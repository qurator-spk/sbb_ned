import pandas as pd
import numpy as np
import os
import torch
import logging
import sqlite3
from multiprocessing import Semaphore

from tqdm import tqdm as tqdm
# noinspection PyUnresolvedReferences
from annoy import AnnoyIndex
from .embeddings.base import EmbedTask, EmbedWithContext, get_embedding_vectors
import json
from qurator.utils.parallel import run as prun
import haversine as hs

from datetime import datetime

logger = logging.getLogger(__name__)


class LookUpByEmbeddings:

    index = None
    entities = None
    entities_file = None
    entity_types = None
    mapping = None
    frequency = None
    n_trees = None
    distance_measure = None
    output_path = None

    init_sem = None

    def __init__(self, page_title, entity_embeddings, embedding_config,
                 entity_title, entity_type, split_parts, max_candidates, search_k, max_dist, context, **kwargs):

        self._entity_embeddings = entity_embeddings
        self._embedding_config = embedding_config
        self._entity_title = entity_title
        self._entity_type = entity_type
        self._page_title = page_title
        self._split_parts = split_parts
        self._max_candidates = max_candidates
        self._search_k = search_k
        self._max_dist = max_dist
        self._context = context

    def __call__(self, *args, **kwargs):

        def get_index_and_mapping(dims):

            self.init_indices(dims)

            return LookUpByEmbeddings.index[self._entity_type], LookUpByEmbeddings.mapping[self._entity_type], \
                LookUpByEmbeddings.frequency[self._entity_type]

        ranking, _ = best_matches(self._entity_embeddings, get_index_and_mapping, self._search_k, self._max_dist)

        before_filter = len(ranking)

        if before_filter < self._search_k:
            logger.warning("LookUpByEmbeddings [{}]: Could only retrieve {} best matches instead of {}".
                         format(self._entity_title, before_filter, self._search_k))

        ranking = ranking.drop_duplicates('guessed_title')

        ranking['on_page'] = self._page_title

        if LookUpByEmbeddings.entities is not None:

            ranking = ranking.merge(LookUpByEmbeddings.entities[['proba']], left_on="guessed_title", right_index=True)

            if self._context is not None and self._entity_type in ['LOC', 'ORG'] and "geographic" in self._context:

                ranking = ranking.merge(LookUpByEmbeddings.entities[['longitude', 'latitude']],
                                        left_on="guessed_title", right_index=True)

                geo_context = self._context["geographic"]
                if type(geo_context) == str:
                    geo_context = [geo_context]

                geo_pos = \
                    LookUpByEmbeddings.entities.loc[LookUpByEmbeddings.entities.QID.isin(geo_context)].copy()

                def make_position(x):
                    try:
                        return float(x.latitude), float(x.longitude)
                    except TypeError:
                        return np.nan

                geo_pos['position'] = 0
                geo_pos['position'] = geo_pos.apply(make_position, axis=1)

                ranking['position'] = 0
                ranking['position'] = ranking.apply(make_position, axis=1)

                def haver(a, b):
                    try:
                        return hs.haversine(a, b)
                    except TypeError:
                        return np.nan

                def geo_dist(r):

                    return geo_pos.apply(lambda g: haver(r.position, g.position), axis=1).min()

                ranking['geo_dist'] = ranking.apply(lambda r: geo_dist(r), axis=1)

                ranking.sort_values(['match_uniqueness', 'dist', 'geo_dist', 'proba', 'match_coverage', 'len_guessed'],
                                    ascending=[False, True, True, False, True, True])

                ranking = ranking.drop(columns=['position', 'longitude', 'latitude', 'geo_dist'])

            elif self._context is not None and "time" in self._context and \
                    'not_after_datetime' in self._context['time']:

                not_after = pd.to_datetime(self._context['time']['not_after_datetime'], utc=True)

                ranking = ranking.merge(LookUpByEmbeddings.entities[['dateofbirth', 'inception']],
                                        left_on="guessed_title", right_index=True)

                ranking['dateofbirth'] = pd.to_datetime(ranking.dateofbirth, yearfirst=True, errors="coerce", utc=True)
                ranking['inception'] = pd.to_datetime(ranking.inception, yearfirst=True, errors="coerce", utc=True)

                ranking = ranking.loc[(ranking.inception.isnull() & ranking.dateofbirth.isnull()) |
                                      (ranking.inception < not_after) | (ranking.dateofbirth < not_after)]

                ranking = ranking.drop(columns=['dateofbirth', 'inception'])

                if len(ranking) == 0:
                    logger.warning("LookUpByEmbeddings [{}]: Not after excludes all of {} candidates".
                                   format(self._entity_title, before_filter))

                ranking = ranking. \
                    sort_values(['match_uniqueness', 'dist', 'proba', 'match_coverage', 'len_guessed'],
                                ascending=[False, True, False, True, True])

            elif self._context is not None and "time" in self._context and \
                    'not_after_year' in self._context['time']:

                not_after_year = self._context['time']['not_after_year']

                ranking = ranking.merge(LookUpByEmbeddings.entities[['dateofbirth']],
                                        left_on="guessed_title", right_index=True)

                ranking = ranking.merge(LookUpByEmbeddings.entities[['inception']],
                                        left_on="guessed_title", right_index=True)

                ranking['yearofinception'] = ranking.inception.apply(
                    lambda x: x if pd.isnull(x) else datetime.strptime(x.split(" ")[0], "%Y-%m-%d").year)

                ranking['yearofbirth'] = ranking.dateofbirth.apply(
                    lambda x: x if pd.isnull(x) else datetime.strptime(x.split(" ")[0], "%Y-%m-%d").year)

                ranking = ranking.loc[(ranking.dateofbirth.isnull()) |
                                      (ranking.yearofbirth < not_after_year)]

                ranking = ranking.loc[(ranking.inception.isnull()) |
                                      (ranking.yearofinception < not_after_year)]

                if len(ranking) == 0:
                    logger.warning("LookUpByEmbeddings [{}]: Not after year excludes all of {} candidates".
                                   format(self._entity_title, before_filter))

                ranking = ranking. \
                    sort_values(['match_uniqueness', 'dist', 'proba', 'match_coverage', 'len_guessed'],
                                ascending=[False, True, False, True, True])

                ranking = ranking.drop(columns=['dateofbirth', 'yearofbirth'])

                ranking = ranking.drop(columns=['inception', 'yearofinception'])

            elif self._context is not None and "time" in self._context and self._entity_type in ['PER'] and \
                    'birth_not_after_year' in self._context['time']:

                not_after_year = self._context['time']['birth_not_after_year']

                ranking = ranking.merge(LookUpByEmbeddings.entities[['dateofbirth']],
                                        left_on="guessed_title", right_index=True)

                ranking['yearofbirth'] = ranking.dateofbirth.apply(
                    lambda x: x if pd.isnull(x) else datetime.strptime(x.split(" ")[0], "%Y-%m-%d").year)

                ranking = ranking.loc[(ranking.dateofbirth.isnull()) |
                                      (ranking.yearofbirth < not_after_year)]

                if len(ranking) == 0:
                    logger.warning("LookUpByEmbeddings [{}]: Not after year excludes all of {} candidates".
                                   format(self._entity_title, before_filter))

                ranking = ranking. \
                    sort_values(['match_uniqueness', 'dist', 'proba', 'match_coverage', 'len_guessed'],
                                ascending=[False, True, False, True, True])

                ranking = ranking.drop(columns=['dateofbirth', 'yearofbirth'])

            else:
                ranking = ranking. \
                    sort_values(['match_uniqueness', 'dist', 'proba', 'match_coverage', 'len_guessed'],
                                ascending=[False, True, False, True, True])
        else:
            ranking = ranking.sort_values(['match_uniqueness', 'dist', 'match_coverage', 'len_guessed'],
                                          ascending=[False, True, False, True])

        ranking = ranking.reset_index(drop=True)

        if self._max_candidates is not None:
            ranking = ranking.iloc[0:self._max_candidates]

        return self._entity_title, ranking

    def init_indices(self, dims):

        if LookUpByEmbeddings.index is not None:
            return

        with LookUpByEmbeddings.init_sem:

            if LookUpByEmbeddings.index is not None:
                return

            LookUpByEmbeddings.index = dict()
            LookUpByEmbeddings.mapping = dict()
            LookUpByEmbeddings.frequency = dict()

            for ent_type in LookUpByEmbeddings.entity_types:

                self._embedding_config['dims'] = dims

                LookUpByEmbeddings.index[ent_type], LookUpByEmbeddings.mapping[ent_type], \
                    LookUpByEmbeddings.frequency[ent_type] = \
                    load(LookUpByEmbeddings.entities_file, self._embedding_config, ent_type, LookUpByEmbeddings.n_trees,
                         LookUpByEmbeddings.distance_measure, LookUpByEmbeddings.output_path)

    @staticmethod
    def initialize(entities_file, entity_types, n_trees, distance_measure, output_path,
                   ned_sql_file=None):

        if ned_sql_file is not None:
            with sqlite3.connect(ned_sql_file) as con:
                LookUpByEmbeddings.entities = pd.read_sql('select * from entities', con=con).\
                    set_index('page_title').\
                    sort_index()

        LookUpByEmbeddings.entities_file = entities_file
        LookUpByEmbeddings.entity_types = entity_types

        LookUpByEmbeddings.n_trees = n_trees
        LookUpByEmbeddings.distance_measure = distance_measure
        LookUpByEmbeddings.output_path = output_path

        LookUpByEmbeddings.init_sem = Semaphore(1)


class LookUpBySurface:

    index = None
    entities_file = None
    embeddings = None
    embedding_conf = None
    mapping = None
    frequency = None
    n_trees = None
    distance_measure = None
    output_path = None
    search_k = None
    max_dist = None

    init_sem = None

    def __init__(self, page_title, entity_surface_parts, entity_title, entity_type, split_parts, max_candidates=None):

        self._entity_surface_parts = entity_surface_parts
        self._entity_title = entity_title
        self._entity_type = entity_type
        self._page_title = page_title
        self._split_parts = split_parts
        self._max_candidates = max_candidates

    def __call__(self, *args, **kwargs):

        def get_index_and_mapping(dims):

            LookUpBySurface.init_indices(dims)

            return LookUpBySurface.index[self._entity_type], LookUpBySurface.mapping[self._entity_type], \
                LookUpBySurface.frequency[self._entity_type]

        rankings = []
        for surface in self._entity_surface_parts:
            text_embeddings = get_embedding_vectors(LookUpBySurface.embeddings[self._entity_type],
                                                    surface, self._split_parts)

            ranking, _ = best_matches(text_embeddings, get_index_and_mapping,
                                      LookUpBySurface.search_k, LookUpBySurface.max_dist)

            ranking['surface'] = surface
            rankings.append(ranking)

        rankings = pd.concat(rankings). \
            drop_duplicates('guessed_title'). \
            sort_values(['match_uniqueness', 'dist', 'match_coverage', 'len_guessed'],
                        ascending=[False, True, False, True]).reset_index(drop=True)

        rankings['on_page'] = self._page_title

        if self._max_candidates is not None:

            rankings = rankings.iloc[0:self._max_candidates]

        return self._entity_title, rankings

    @staticmethod
    def _get_all(data_sequence, ent_types, split_parts, sem=None):

        for _, article in data_sequence:

            sentences = json.loads(article.text)
            sen_link_titles = json.loads(article.link_titles)
            sen_tags = json.loads(article.tags)

            if len(ent_types.intersection({t if len(t) < 3 else t[2:] for tags in sen_tags for t in tags})) == 0:
                # Do not further process sentences that do not contain a relevant linked entity of type "ent_types".
                continue

            for sen, link_titles, tags in zip(sentences, sen_link_titles, sen_tags):

                if len(ent_types.intersection({t if len(t) < 3 else t[2:] for t in tags})) == 0:
                    # Do not further process sentences that do not contain a relevant linked entity of type "ent_types".
                    continue

                entity_surface_parts = []
                entity_title = ''
                entity_type = None
                for word, link_title, tag in zip(sen, link_titles, tags):

                    if (tag == 'O' or tag.startswith('B-')) and len(entity_surface_parts) > 0:

                        if sem is not None:
                            sem.acquire(timeout=10)

                        yield LookUpBySurface(article.page_title, entity_surface_parts, entity_title, entity_type,
                                              split_parts)

                        entity_surface_parts = []

                    if tag != 'O' and tag[2:] in ent_types:

                        entity_surface_parts.append(word)
                        entity_title = link_title
                        entity_type = tag[2:]

                if len(entity_surface_parts) > 0:

                    if sem is not None:
                        sem.acquire(timeout=10)

                    yield LookUpBySurface(article.page_title, entity_surface_parts, entity_title, entity_type,
                                          split_parts)

    @staticmethod
    def run(entities_file, embeddings, data_sequence, split_parts, processes, n_trees, distance_measure, output_path,
            search_k, max_dist, sem=None):

        return prun(LookUpBySurface._get_all(data_sequence, set(embeddings.keys()), split_parts, sem=sem),
                    processes=processes,
                    initializer=LookUpBySurface.initialize,
                    initargs=(entities_file, embeddings, n_trees, distance_measure, output_path, search_k, max_dist))

    @staticmethod
    def init_indices(dims):

        if LookUpBySurface.index is not None:
            return

        with LookUpBySurface.init_sem:

            if LookUpBySurface.index is not None:
                return

            LookUpBySurface.index = dict()
            LookUpBySurface.mapping = dict()
            LookUpBySurface.frequency = dict()

            for ent_type, emb in LookUpBySurface.embeddings.items():

                config = emb.config()
                config['dims'] = dims

                LookUpBySurface.index[ent_type], LookUpBySurface.mapping[ent_type], LookUpBySurface.frequency[ent_type] = \
                    load(LookUpBySurface.entities_file, config, ent_type, LookUpBySurface.n_trees,
                         LookUpBySurface.distance_measure, LookUpBySurface.output_path)

    @staticmethod
    def initialize(entities_file, embeddings, n_trees, distance_measure, output_path, search_k, max_dist):

        LookUpBySurface.entities_file = entities_file
        LookUpBySurface.embeddings = dict()

        for ent_type, emb in embeddings.items():

            if type(emb) == tuple:
                LookUpBySurface.embeddings[ent_type] = emb[0](**emb[1])
            else:
                LookUpBySurface.embeddings[ent_type] = emb

        LookUpBySurface.n_trees = n_trees
        LookUpBySurface.distance_measure = distance_measure
        LookUpBySurface.output_path = output_path
        LookUpBySurface.search_k = search_k
        LookUpBySurface.max_dist = max_dist

        LookUpBySurface.init_sem = Semaphore(1)


class LookUpBySurfaceAndContext:

    index = None
    index_file = None
    mapping = None
    search_k = None
    distance_measure = None
    init_sem = None

    def __init__(self, link_result):

        self._link_result = link_result

    def __call__(self, *args, **kwargs):

        e = self._link_result.drop(['entity_title', 'count']).astype(np.float32).values
        e /= float(self._link_result['count'])

        LookUpBySurfaceAndContext.init_index(len(e))

        ann_indices, dist = LookUpBySurfaceAndContext.index.get_nns_by_vector(e, LookUpBySurfaceAndContext.search_k,
                                                                              include_distances=True)

        ranking = LookUpBySurfaceAndContext.mapping.loc[ann_indices]
        ranking['dist'] = dist

        ranking = ranking.sort_values('dist', ascending=True).reset_index(drop=True).reset_index(). \
            rename(columns={'index': 'rank', 'page_title': 'guessed_title'})

        success = ranking.loc[ranking.guessed_title == self._link_result.entity_title]

        if len(success) > 0:
            return self._link_result.entity_title, success
        else:
            return self._link_result.entity_title, ranking.iloc[[0]]

    @staticmethod
    def init_index(dims):

        if LookUpBySurfaceAndContext.index is not None:
            return

        LookUpBySurfaceAndContext.init_sem.aqcuire()

        if LookUpBySurfaceAndContext.index is not None:
            return

        LookUpBySurfaceAndContext.index = AnnoyIndex(dims, LookUpBySurfaceAndContext.distance_measure)

        LookUpBySurfaceAndContext.index.load(LookUpBySurfaceAndContext.index_file)

        LookUpBySurfaceAndContext.init_sem.release()

        return

    @staticmethod
    def initialize(index_file, mapping_file, distance_measure, search_k):

        LookUpBySurfaceAndContext.search_k = search_k
        LookUpBySurfaceAndContext.index_file = index_file
        LookUpBySurfaceAndContext.distance_measure = distance_measure

        LookUpBySurfaceAndContext.init_sem = Semaphore(1)

        LookUpBySurfaceAndContext.mapping = pd.read_pickle(mapping_file)

    @staticmethod
    def _get_all(embeddings, data_sequence, start_iteration, ent_type, w_size, batch_size, processes,
                 evalutation_semaphore=None):

        # The embed semaphore makes sure that the EmbedWithContext will not over produce results in relation
        # to the LookUpBySurfaceAndContext creation
        embed_semaphore = Semaphore(100)

        for it, link_result in \
                enumerate(
                    EmbedWithContext.run(embeddings, data_sequence, ent_type, w_size, batch_size,
                                         processes, embed_semaphore, start_iteration=start_iteration)):
            try:
                if evalutation_semaphore is not None:
                    evalutation_semaphore.acquire(timeout=10)

                yield LookUpBySurfaceAndContext(link_result)

            except Exception as ex:
                print(type(ex))
                print("Error: ", link_result)
                raise

            if it % batch_size == 0:
                embed_semaphore.release()

    @staticmethod
    def run(index_file, mapping_file, distance_measure, search_k,
            embeddings, data_sequence, start_iteration, ent_type, w_size, batch_size, processes, sem=None):

        return prun(
            LookUpBySurfaceAndContext._get_all(embeddings, data_sequence, start_iteration, ent_type, w_size, batch_size,
                                               processes, sem), processes=3*processes,
            initializer=LookUpBySurfaceAndContext.initialize,
            initargs=(index_file, mapping_file, distance_measure, search_k))


def index_file_name(embedding_description, ent_type, n_trees, distance_measure):
    return 'index-n_trees_{}-dist_{}-emb_{}-{}.ann'.format(n_trees, distance_measure, embedding_description, ent_type)


def mapping_file_name(embedding_description, ent_type, n_trees, distance_measure):
    return 'mapping-n_trees_{}-dist_{}-emb_{}-{}.pkl'.format(n_trees, distance_measure, embedding_description, ent_type)


def build(all_entities_file, embeddings, ent_type, n_trees, processes=10, distance_measure='angular', split_parts=True,
          path='.', max_iter=None):

    all_entities = pd.read_pickle(all_entities_file)

    if ent_type in all_entities.columns:

        all_entities = all_entities.loc[all_entities[ent_type], ['TYPE', 'label']]
        all_entities['TYPE'] = ent_type

    all_entities = all_entities.loc[all_entities.TYPE == ent_type]

    prefix = ".".join(os.path.basename(all_entities_file).split('.')[:-1])

    # wiki_index is an approximate nearest neighbour index that permits fast lookup of an ann_index for some
    # given embedding vector. The ann_index than points to a number of entities according to a mapping (see below).
    index = None  # lazy creation

    # mapping provides a map from ann_index (approximate nearest neighbour index) -> entity (title)
    # That mapping is not unique, i.e., a particular ann_index might point to many different entities.
    mapping = []

    # part_dict temporarily stores those part/ann_index pairs that have already been added to mapping
    part_dict = dict()

    ann_index = 0
    for idx, (title, embeded, _) in enumerate(EmbedTask.run(embeddings, all_entities, split_parts, processes)):

        if max_iter is not None and idx > max_iter:
            break

        if idx % 1000 == 0:
            torch.cuda.empty_cache()

        for part, part_embedding in embeded.iterrows():

            if part in part_dict:
                mapping.append((part_dict[part], title))
            else:
                if index is None:
                    index = AnnoyIndex(len(part_embedding), distance_measure)

                part_dict[part] = ann_index

                index.add_item(ann_index, part_embedding)

                mapping.append((ann_index, title))
                ann_index += 1

    mapping = pd.DataFrame(mapping, columns=['ann_index', 'page_title'])
    mapping['num_parts'] = mapping.loc[:, 'page_title'].str.split(" |-|_").str.len()

    if type(embeddings) == tuple:
        embeddings = embeddings[0](**embeddings[1])

    mapping.to_pickle("{}/{}-{}".format(path, prefix,
                                        mapping_file_name(embeddings.description(), ent_type=ent_type, n_trees=n_trees,
                                                       distance_measure=distance_measure)))
    del mapping

    index.build(n_trees)

    index.save("{}/{}-{}".format(path, prefix, index_file_name(embeddings.description(), ent_type=ent_type,
                                                               n_trees=n_trees, distance_measure=distance_measure)))


def build_from_matrix(context_matrix_file, distance_measure, n_trees):

    result_file = "{}-dm_{}-nt_{}.ann".format(".".join(context_matrix_file.split('.')[:-1]), distance_measure, n_trees)

    mapping_file = "{}-dm_{}-nt_{}.mapping".format(".".join(context_matrix_file.split('.')[:-1]), distance_measure,
                                                   n_trees)

    print("\n\n\n write result to {} and {}".format(result_file, mapping_file))

    cm = pd.read_pickle(context_matrix_file)

    index = AnnoyIndex(cm.shape[1] - 1, distance_measure)

    for r_idx, (page_title, row) in tqdm(enumerate(cm.iterrows()), total=len(cm)):

        vec = row.iloc[1:].values
        count = row.iloc[0]

        index.add_item(r_idx, vec/count)

    pd.DataFrame(index=cm.index).reset_index().to_pickle(mapping_file)
    del cm

    index.build(n_trees)

    index.save(result_file)


def load(entities_file, embedding_config, ent_type, n_trees, distance_measure='angular', path='.',
         min_title_len=3):

    prefix = ".".join(os.path.basename(entities_file).split('.')[:-1])

    filename = "{}/{}-{}".format(path, prefix,
                                 index_file_name(embedding_config['description'], n_trees=n_trees,
                                                 distance_measure=distance_measure, ent_type=ent_type))

    index = AnnoyIndex(embedding_config['dims'], distance_measure)

    index.load(filename)

    mapping = pd.read_pickle("{}/{}-{}".format(path, prefix,
                                               mapping_file_name(embedding_config['description'], n_trees=n_trees,
                                                                 distance_measure=distance_measure, ent_type=ent_type)))

    # filter out those index entries that link to more than max_occurences different entities
    vc = mapping.ann_index.value_counts()

    # mapping = mapping.loc[~mapping.ann_index.isin(vc.loc[vc > max_occurences].index)]

    mapping = mapping.loc[~((mapping.page_title.str.len() < min_title_len) | (mapping.page_title.str.isnumeric()))]

    mapping = mapping.set_index('ann_index').sort_index()

    return index, mapping, vc


def best_matches(text_embeddings, get_index_and_mapping, search_k, max_dist=0.25, summarizer='max', min_part_len=4,
                 max_frequency=1000):

    hits = []
    ranking = []
    lookup = []
    mapping = None
    frequency = None
    index = None

    for part, e in text_embeddings.iterrows():

        if mapping is None:
            index, mapping, frequency = get_index_and_mapping(len(e))

        ann_indices, dist = index.get_nns_by_vector(e, search_k, include_distances=True)

        if len(part) < min_part_len:

            valid = [i for i in range(0, len(ann_indices)) if frequency[ann_indices[i]] <= max_frequency]

            ann_indices = [ann_indices[i] for i in valid]
            dist = [dist[i] for i in valid]

        lookup.append(pd.DataFrame({'ann_index': ann_indices, 'dist': dist, 'part': len(dist)*[part]}))

    if len(lookup) > 0:

        lookup = pd.concat(lookup).sort_values('ann_index')

        hits = mapping.loc[mapping.index.isin(lookup.ann_index)].copy()

        hits = hits.merge(lookup, left_on="ann_index", right_on='ann_index')

    if len(hits) >= 1:

        hits = hits.loc[hits.dist < max_dist]

        hit_counter = hits.part.value_counts()

        for guessed_title, matched in hits.groupby('page_title', as_index=False):

            matched = matched.drop_duplicates(subset=['part'])

            match_uniqueness = (float(len(hits)) / hit_counter[matched.part]).sum() * len(matched)

            match_coverage = float(len(guessed_title))/matched.part.astype(str).str.len().sum()

            summarized_dist_over_all_parts = matched.dist.apply(summarizer)

            ranking.append((guessed_title, summarized_dist_over_all_parts, match_uniqueness, match_coverage,
                            matched.part.iloc[0]))

        ranking = pd.DataFrame(ranking, columns=['guessed_title', 'dist', 'match_uniqueness', 'match_coverage',
                                                 'surface'])

        ranking['len_guessed'] = ranking.guessed_title.str.len()

        ranking = ranking.sort_values(['match_uniqueness', 'dist', 'match_coverage', 'len_guessed'],
                                      ascending=[False, True, True, True]).reset_index(drop=True)

    if len(ranking) == 0:
        ranking = \
            pd.DataFrame({'guessed_title': '', 'dist': np.inf, 'match_uniqueness': 0, 'match_coverage': 0,
                          'len_guessed': np.inf}, index=[0])
        hits = None

    return ranking, hits

