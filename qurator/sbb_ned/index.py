import pandas as pd
import numpy as np
from multiprocessing import Semaphore

from tqdm import tqdm as tqdm
# noinspection PyUnresolvedReferences
from annoy import AnnoyIndex
from .embeddings.base import EmbedTask, EmbedWithContext, get_embedding_vectors
import json
from qurator.utils.parallel import run as prun


class LookUpBySurface:

    index = None
    embeddings = None
    mapping = None
    search_k = None
    max_dist = None

    def __init__(self, page_title, entity_surface_parts, entity_title, entity_type, split_parts):

        self._entity_surface_parts = entity_surface_parts
        self._entity_title = entity_title
        self._entity_type = entity_type
        self._page_title = page_title
        self._split_parts = split_parts

    def __call__(self, *args, **kwargs):

        surface_text = " ".join(self._entity_surface_parts)

        text_embeddings = get_embedding_vectors(LookUpBySurface.embeddings[self._entity_type], surface_text,
                                                self._split_parts)

        ranking, hits = best_matches(text_embeddings, LookUpBySurface.index[self._entity_type],
                                     LookUpBySurface.mapping[self._entity_type],
                                     LookUpBySurface.search_k, LookUpBySurface.max_dist)

        ranking['on_page'] = self._page_title
        ranking['surface'] = surface_text

        return self._entity_title, ranking

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
    def run(embeddings, data_sequence, split_parts, processes, n_trees, distance_measure, output_path,
            search_k, max_dist, sem=None):

        return prun(LookUpBySurface._get_all(data_sequence, set(embeddings.keys()), split_parts, sem=sem),
                    processes=processes,
                    initializer=LookUpBySurface.initialize,
                    initargs=(embeddings, n_trees, distance_measure, output_path, search_k, max_dist))

    @staticmethod
    def initialize(embeddings, n_trees, distance_measure, output_path, search_k, max_dist):

        LookUpBySurface.embeddings = dict()
        LookUpBySurface.index = dict()
        LookUpBySurface.mapping = dict()

        for ent_type, emb in embeddings.items():

            if type(emb) == tuple:
                LookUpBySurface.embeddings[ent_type] = emb[0](**emb[1])
            else:
                LookUpBySurface.embeddings[ent_type] = emb

            LookUpBySurface.index[ent_type], LookUpBySurface.mapping[ent_type] = \
                load(LookUpBySurface.embeddings[ent_type].config(), ent_type, n_trees, distance_measure, output_path)

        LookUpBySurface.search_k = search_k
        LookUpBySurface.max_dist = max_dist


class LookUpBySurfaceAndContext:

    index = None
    mapping = None
    search_k = None

    def __init__(self, link_result):

        self._link_result = link_result

    def __call__(self, *args, **kwargs):

        e = self._link_result.drop(['entity_title', 'count']).astype(np.float32).values
        e /= float(self._link_result['count'])

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
    def initialize(index_file, mapping_file, dims, distance_measure, search_k):

        LookUpBySurfaceAndContext.search_k = search_k

        LookUpBySurfaceAndContext.index = AnnoyIndex(dims, distance_measure)

        LookUpBySurfaceAndContext.index.load(index_file)

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
    def run(index_file, mapping_file, dims, distance_measure, search_k,
            embeddings, data_sequence, start_iteration, ent_type, w_size, batch_size, processes, sem=None):

        return prun(
            LookUpBySurfaceAndContext._get_all(embeddings, data_sequence, start_iteration, ent_type, w_size, batch_size,
                                               processes, sem), processes=3*processes,
            initializer=LookUpBySurfaceAndContext.initialize,
            initargs=(index_file, mapping_file, dims, distance_measure, search_k))


def index_file_name(embedding_description, ent_type, n_trees, distance_measure):
    return 'title-index-n_trees_{}-dist_{}-emb_{}-{}.ann'.format(n_trees, distance_measure,
                                                                 embedding_description, ent_type)


def mapping_file_name(embedding_description, ent_type, n_trees, distance_measure):
    return 'title-mapping-n_trees_{}-dist_{}-emb_{}-{}.pkl'.format(n_trees, distance_measure,
                                                                   embedding_description, ent_type)


def build(all_entities, embeddings, dims, ent_type, n_trees, processes=10, distance_measure='angular', split_parts=True,
          path='.'):

    all_entities = all_entities.loc[all_entities.TYPE == ent_type]

    # wiki_index is an approximate nearest neighbour index that permits fast lookup of an ann_index for some
    # given embedding vector. The ann_index than points to a number of entities according to a mapping (see below).
    index = AnnoyIndex(dims, distance_measure)

    # mapping provides a map from ann_index (approximate nearest neighbour index) -> entity (title)
    # That mapping is not unique, i.e., a particular ann_index might point to many different entities.
    mapping = []

    # part_dict temporarily stores those part/ann_index pairs that have already been added to mapping
    part_dict = dict()

    ann_index = 0
    for title, embeddings in EmbedTask.run(embeddings, all_entities, split_parts, processes):

        for part, part_embedding in embeddings.iterrows():

            if part in part_dict:
                mapping.append((part_dict[part], title))
            else:
                part_dict[part] = ann_index

                index.add_item(ann_index, part_embedding)

                mapping.append((ann_index, title))
                ann_index += 1

    mapping = pd.DataFrame(mapping, columns=['ann_index', 'page_title'])
    mapping['num_parts'] = mapping.loc[:, 'page_title'].str.split(" |-|_").str.len()

    mapping.to_pickle("{}/{}".format(path, mapping_file_name(embeddings.description(), ent_type, n_trees,
                                                             distance_measure)))
    del mapping

    index.build(n_trees)

    index.save("{}/{}".format(path, index_file_name(embeddings.description(), ent_type, n_trees, distance_measure)))


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


def load(embedding_config, ent_type, n_trees, distance_measure='angular', path='.', max_occurences=1000):

    index = AnnoyIndex(embedding_config['dims'], distance_measure)

    index.load("{}/{}".format(path, index_file_name(embedding_config['description'],
                                                    n_trees, distance_measure, ent_type)))

    mapping = pd.read_pickle("{}/{}".format(path, mapping_file_name(embedding_config['description'],
                                                                    n_trees, distance_measure, ent_type)))

    # filter out those index entries that link to more than max_occurences different entities
    vc = mapping.ann_index.value_counts()
    mapping = mapping.loc[~mapping.ann_index.isin(vc.loc[vc > max_occurences].index)]

    mapping = mapping.set_index('ann_index').sort_index()

    return index, mapping


def best_matches(text_embeddings, index, mapping, search_k=10, max_dist=0.25, summarizer='max'):

    hits = []

    for part, e in text_embeddings.iterrows():

        ann_indices, dist = index.get_nns_by_vector(e, search_k, include_distances=True)

        lookup_index = pd.DataFrame({'ann_index': ann_indices, 'dist': dist})

        related_pages = mapping.loc[mapping.index.isin(ann_indices)].copy()

        related_pages = related_pages.merge(lookup_index, left_index=True, right_on='ann_index')
        related_pages['part'] = part

        hits.append(related_pages)

    if len(hits) < 1:
        ranking = pd.DataFrame({'guessed_title': '', 'dist': np.inf, 'len_pa': 0, 'rank': -1}, index=[0])

        hits = None
    else:
        hits = pd.concat(hits)

        hits = hits.loc[hits.dist < max_dist]

        ranking = []

        for page_title, matched in hits.groupby('page_title', as_index=False):

            weighted = (len(hits)/hits.part.value_counts()[matched.drop_duplicates(subset=['part']).part]).sum()

            num_matched_parts = len(matched.drop_duplicates(subset=['part']))

            # weighted = factor * len(matched.drop_duplicates(subset=['part']))

            # if len(matched) >= 3:
            #    import ipdb;ipdb.set_trace()

            summarized_dist_over_all_parts = matched.dist.apply(summarizer)

            ranking.append((page_title, summarized_dist_over_all_parts, weighted))

        ranking = pd.DataFrame(ranking, columns=['guessed_title', 'dist', 'len_pa'])

        ranking['len_guessed'] = ranking.guessed_title.str.len()

        ranking = ranking.sort_values(['len_pa', 'dist', 'len_guessed'],
                                      ascending=[False, True, True]).reset_index(drop=True)

        ranking['rank'] = ranking.index

    if len(ranking) == 0:
        ranking = pd.DataFrame({'guessed_title': '', 'dist': np.inf, 'len_pa': 0, 'rank': -1}, index=[0])

    return ranking, hits
