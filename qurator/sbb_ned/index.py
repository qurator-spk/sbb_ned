import pandas as pd
import numpy as np

from tqdm import tqdm as tqdm
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


def get_embedding_vectors(embeddings, text):
    parts = [re.sub(r'[\W_]+', '', p) for p in re.split(" |-|_", text)]
    vectors = []
    for p in parts:
        vectors.append(embeddings.get(p).astype(np.float32))

    return pd.DataFrame(vectors, index=parts)


class EmbedTask:
    embeddings = None

    def __init__(self, index, title):
        self._index = index
        self._title = title

    def __call__(self, *args, **kwargs):
        return {'title': self._title,
                'embeddings': get_embedding_vectors(EmbedTask.embeddings, self._title)}

    @staticmethod
    def initialize(embeddings):
        EmbedTask.embeddings = embeddings


def index_file_name(embeddings, ent_type, n_trees, distance_measure):
    return 'title-index-n_trees_{}-dist_{}-emb_{}-{}.ann'.format(n_trees, distance_measure,
                                                                 embeddings.description(), ent_type)


def mapping_file_name(embeddings, ent_type, n_trees, distance_measure):
    return 'title-mapping-n_trees_{}-dist_{}-emb_{}-{}.pkl'.format(n_trees, distance_measure,
                                                                   embeddings.description(), ent_type)


def get_embed_tasks(all_entities):
    for i, (title, v) in tqdm(enumerate(all_entities.iterrows()), total=len(all_entities)):
        yield EmbedTask(i, title)


def build(all_entities, embeddings, ent_type, n_trees, processes=10, distance_measure='angular', path='.'):

    wiki_index = AnnoyIndex(embeddings.dims(), distance_measure)
    mapping = []
    part_dict = dict()

    ann_index = 0
    for res in run(get_embed_tasks(all_entities.loc[all_entities.TYPE == ent_type]), processes=processes,
                   initializer=EmbedTask.initialize, initargs=(embeddings,)):

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
    wiki_index.save("{}/{}".format(path, index_file_name(embeddings, n_trees, distance_measure, ent_type)))

    mapping = pd.DataFrame(mapping, columns=['ann_index', 'page_title'])
    mapping['num_parts'] = mapping.loc[:, 'page_title'].str.split(" |-|_").str.len()

    mapping.to_pickle("{}/{}".format(path, mapping_file_name(embeddings, n_trees, distance_measure, ent_type)))


def load(embeddings, ent_type, n_trees, distance_measure='angular', path='.'):

    index = AnnoyIndex(embeddings.dims(), distance_measure)

    index.load("{}/{}".format(path, index_file_name(embeddings, n_trees, distance_measure, ent_type)))

    mapping = pd.read_pickle("{}/{}".format(path,
                                            mapping_file_name(embeddings, n_trees, distance_measure, ent_type)))

    return index, mapping


def best_matches(text, index, embeddings, mapping, search_k=10, max_dist=0.25):

    text_embeddings = get_embedding_vectors(embeddings, text)

    hits = []

    for part, e in text_embeddings.iterrows():

        ann_indices, dist = index.get_nns_by_vector(e, search_k, include_distances=True)

        lookup_index = pd.DataFrame({'ann_index': ann_indices, 'dist': dist})

        lookup_mapping = mapping.loc[mapping['ann_index'].isin(ann_indices)].copy()

        lookup_mapping = lookup_mapping.merge(lookup_index, left_on="ann_index", right_on='ann_index')
        lookup_mapping['part'] = part

        hits.append(lookup_mapping)

    hits = pd.concat(hits)

    hits = hits.loc[hits.dist < max_dist]

    ranking = []

    for page_title, matched in hits.groupby('page_title', as_index=False):

        rank = 0.0

        for _, match_group in matched.groupby('ann_index'):
            rank += match_group.dist.min()

        ranking.append((page_title, rank, len(matched)))

    ranking = pd.DataFrame(ranking, columns=['page_title', 'rank', 'len_pa']).sort_values('rank')

    return ranking, hits
