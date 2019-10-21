import pandas as pd
import numpy as np

from tqdm import tqdm as tqdm
# noinspection PyUnresolvedReferences
from annoy import AnnoyIndex
import re

from qurator.utils.parallel import run


class Embeddings:

    def __init__(self, *args, **kwargs):
        pass

    def dims(self):
        raise NotImplementedError()

    def get(self, key):
        raise NotImplementedError()

    def description(self):
        raise NotImplementedError()


def get_embedding_vectors(embeddings, text, split_parts):

    if split_parts:
        parts = [re.sub(r'[\W_]+', '', p) for p in re.split(" |-|_", text)]
        parts = [p for p in parts if len(p) > 0]
    else:
        parts = [text]

    vectors = []
    vector_parts = []
    for vp, emb in embeddings.get(parts):

        vector_parts.append(vp)
        vectors.append(emb.astype(np.float32))

    ret = pd.DataFrame(vectors, index=vector_parts)

    return ret


class EmbedTask:
    embeddings = None

    def __init__(self, index, title, split_parts):
        self._index = index
        self._title = title
        self._split_parts = split_parts

    def __call__(self, *args, **kwargs):
        return {'title': self._title,
                'embeddings': get_embedding_vectors(EmbedTask.embeddings, self._title, self._split_parts)}

    @staticmethod
    def initialize(embeddings):
        EmbedTask.embeddings = embeddings


def index_file_name(embedding_description, ent_type, n_trees, distance_measure):
    return 'title-index-n_trees_{}-dist_{}-emb_{}-{}.ann'.format(n_trees, distance_measure,
                                                                 embedding_description, ent_type)


def mapping_file_name(embedding_description, ent_type, n_trees, distance_measure):
    return 'title-mapping-n_trees_{}-dist_{}-emb_{}-{}.pkl'.format(n_trees, distance_measure,
                                                                   embedding_description, ent_type)


def get_embed_tasks(all_entities, split_parts):
    for i, (title, v) in tqdm(enumerate(all_entities.iterrows()), total=len(all_entities)):
        yield EmbedTask(i, title, split_parts)


def build(all_entities, embeddings, ent_type, n_trees, processes=10, distance_measure='angular', split_parts=True,
          path='.'):

    # wiki_index is an approximate nearest neighbour index that permits fast lookup of an ann_index for some
    # given embedding vector. The ann_index than points to a number of entities according to a mapping (see below).
    wiki_index = AnnoyIndex(embeddings.dims(), distance_measure)

    # mapping provides a map from ann_index (approximate nearest neighbour index) -> entity (title)
    # That mapping is not unique, i.e., a particular ann_index might point to many different entities.
    mapping = []

    # part_dict temporarily stores those part/ann_index pairs that have already been added to mapping
    part_dict = dict()

    ann_index = 0
    for res in run(get_embed_tasks(all_entities.loc[all_entities.TYPE == ent_type], split_parts), processes=processes,
                   initializer=EmbedTask.initialize, initargs=(embeddings,)):

        if 'embeddings' in locals():
            del embeddings

        title = res['title']

        for part, e in res['embeddings'].iterrows():

            if part in part_dict:
                mapping.append((part_dict[part], title))
            else:
                part_dict[part] = ann_index

                wiki_index.add_item(ann_index, e)

                mapping.append((ann_index, title))
                ann_index += 1

    wiki_index.build(n_trees)

    wiki_index.save("{}/{}".format(path, index_file_name(embeddings.description(), ent_type, n_trees,
                                                         distance_measure)))

    mapping = pd.DataFrame(mapping, columns=['ann_index', 'page_title'])
    mapping['num_parts'] = mapping.loc[:, 'page_title'].str.split(" |-|_").str.len()

    mapping.to_pickle("{}/{}".format(path, mapping_file_name(embeddings.description(), ent_type, n_trees,
                                                             distance_measure)))


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

            num_matched_parts = len(matched)

            summarized_dist_over_all_parts = matched.dist.apply(summarizer)

            ranking.append((page_title, summarized_dist_over_all_parts, num_matched_parts))

        ranking = pd.DataFrame(ranking, columns=['guessed_title', 'dist', 'len_pa']).\
            sort_values(['len_pa', 'dist'], ascending=[False, True]).\
            reset_index(drop=True)

        ranking['rank'] = ranking.index

    if len(ranking) == 0:
        ranking = pd.DataFrame({'guessed_title': '', 'dist': np.inf, 'len_pa': 0, 'rank': -1}, index=[0])

    return ranking, hits
