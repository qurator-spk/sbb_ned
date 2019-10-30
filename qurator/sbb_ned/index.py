import pandas as pd
import numpy as np

from tqdm import tqdm as tqdm
# noinspection PyUnresolvedReferences
from annoy import AnnoyIndex
from .embeddings.base import EmbedTask


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
