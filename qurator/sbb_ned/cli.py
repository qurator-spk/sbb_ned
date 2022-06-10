import warnings
import logging

warnings.filterwarnings('ignore', category=FutureWarning)

from .index import build as build_index
from .index import build_from_matrix # , LookUpBySurface, LookUpBySurfaceAndContext
# from .embeddings.base import load_embeddings , EmbedWithContext
from .embeddings.base import load_embeddings  # , EmbedWithContext
from .ground_truth.data_processor import WikipediaDataset, InputExample, convert_examples_to_features
import click
import numpy as np
import pandas as pd
# import dask.dataframe as dd
from tqdm import tqdm as tqdm

# import pyarrow as pa
# import pyarrow.parquet as pq

# from pathlib import Path
from qurator.utils.parallel import run as prun
# from numpy.linalg import norm
# from numpy.matlib import repmat

import json
import sqlite3

from sklearn.utils import shuffle
from qurator.sbb_ner.models.tokenization import BertTokenizer
from multiprocessing import Semaphore

logger = logging.getLogger(__name__)


@click.command()
@click.argument('all-entities-file', type=click.Path(exists=True), required=True, nargs=1)
@click.argument('embedding-type', type=click.Choice(['fasttext', 'bert', 'flair', 'anagram']), required=True, nargs=1)
@click.argument('entity-type', type=str, required=True, nargs=1)
@click.argument('n-trees', type=int, required=True, nargs=1)
@click.argument('output-path', type=click.Path(exists=True), required=True, nargs=1)
@click.option('--n-processes', type=int, default=6, help='Number of parallel processes. default: 6.')
@click.option('--distance-measure', type=click.Choice(['angular', 'euclidean']), default='angular',
              help="default: angular")
@click.option('--split-parts', type=bool, is_flag=True, help="Process entity surfaces in parts.")
@click.option('--model-path', type=click.Path(exists=True),
              default=None, help="From where to load the embedding model.")
@click.option('--layers', type=str, default="-1,-2,-3,-4", help="Which layers to use. default -1,-2,-3,-4")
@click.option('--pooling', type=str, default="first", help="How to pool the output for different tokens/words. "
                                                           "default: first.")
@click.option('--scalar-mix', type=bool, is_flag=True, help="Use scalar mix of layers.")
@click.option('--max-iter', type=int, default=None, help='Perform only max-iter iterations (for testing purposes). '
                                                         'default: process everything.')
def build(all_entities_file, embedding_type, entity_type, n_trees, output_path,
          n_processes, distance_measure, split_parts, model_path, layers, pooling, scalar_mix=False, max_iter=None):
    """
    Create an approximative nearest neightbour index, based on the surface strings of entities that enables a fast
    lookup of NE-candidates.

    ALL_ENTITIES_FILE: Pandas DataFrame pickle that contains all entites.

    EMBEDDING_TYPE: Type of embedding [ fasttext, bert, flair, anagram ]

    ENTITY_TYPE: Type of entities, for instance ORG, LOC, PER ...

    N_TREES: Number of trees in the approximative nearest neighbour index

    OUTPUT_PATH: Where to write the result files.
    """

    embeddings = load_embeddings(embedding_type, model_path=model_path, layers=layers, pooling_operation=pooling,
                                 use_scalar_mix=scalar_mix)

    build_index(all_entities_file, embeddings, entity_type, n_trees, n_processes, distance_measure, split_parts,
                output_path, max_iter)


# @click.command()
# @click.argument('tagged-parquet', type=click.Path(exists=True), required=True, nargs=1)
# @click.argument('embedding-type', type=click.Choice(['fasttext', 'bert']), required=True, nargs=1)
# @click.argument('entities_file', type=str, required=True, nargs=1)
# @click.argument('ent-type', type=str, required=True, nargs=1)
# @click.argument('n-trees', type=int, required=True, nargs=1)
# @click.argument('distance-measure', type=click.Choice(['angular', 'euclidean']), required=True, nargs=1)
# @click.argument('output-path', type=click.Path(exists=True), required=True, nargs=1)
# @click.option('--search-k', type=int, default=50, help="Number of NN to be considered. default: 50.")
# @click.option('--max-dist', type=float, default=0.25, help="Maximum permitted NN distance. default: 0.25")
# @click.option('--processes', type=int, default=6, help='Number of parallel processes. default: 6.')
# @click.option('--save-interval', type=int, default=10000, help='Save result every N steps. default: 10000.')
# @click.option('--split-parts', type=bool, is_flag=True, help="Process entity surfaces in parts.")
# @click.option('--max-iter', type=float, default=np.inf, help="Number of evaluation iterations. "
#                                                              "default: evaluate everything.")
# @click.option('--model-path', type=click.Path(exists=True),
#               default=None, help="from where to load the embedding model.")
# def evaluate(tagged_parquet, embedding_type, entities_file, ent_type, n_trees,
#              distance_measure, output_path, search_k, max_dist, processes, save_interval,
#              split_parts, max_iter, model_path):
#     """
#     Evaluate the NE-lookup performance of some approximative nearest neighbour index.
#     Runs through a many Wikipedia texts where the occurrences of named entities have been marked.
#     Determines how often the ANN-index manages to provide the correct candidate among the nearest neighbours.
#
#     TAGGET_PARQUET: A sqlite file that contains the pre-processed wikipedia text (see tag_entities2sqlite for details)
#     EMBEDDING_TYPE: 'fasttext' or 'bert'
#     ENTITIES_FILE: The entity table as pickled Pandas DataFrame.
#     ENT_TYPE: What type of entities should be considered, for instance: 'PER', 'LOC' or 'ORG'.
#     N_TREES: Number trees in the approximative nearest neighbour index.
#     DISTANCE_MEASURE: of the approximative nearest neighbour index, i.e, 'angular' or 'euclidian'.
#     OUTPUT_PATH: Where to store the result.
#     """
#
#     embeddings = load_embeddings(embedding_type, model_path=model_path)
#
#     print("Reading entity linking ground-truth file: {}".format(tagged_parquet))
#     df = dd.read_parquet(tagged_parquet)
#     print("done.")
#
#     data_sequence = tqdm(df.iterrows(), total=len(df))
#
#     result_path = '{}/nedstat-embt_{}-entt_{}-nt_{}-dm_{}-sk_{}-md_{}.parquet'.\
#         format(output_path, embedding_type, ent_type, n_trees, distance_measure, search_k, max_dist)
#
#     print("Write result statistics to: {} .".format(result_path))
#
#     total_successes = mean_rank = mean_len_rank = 0
#
#     results = []
#
#     def write_results():
#
#         nonlocal results
#
#         if len(results) == 0:
#             return
#
#         res = pd.concat(results)
#
#         # noinspection PyArgumentList
#         table = pa.Table.from_pandas(res)
#
#         pq.write_to_dataset(table, root_path=result_path)
#
#         results = []
#
#     for total_processed, (entity_title, ranking) in \
#             enumerate(LookUpBySurface.run(entities_file, {ent_type: embeddings}, data_sequence, split_parts, processes,
#                                           n_trees, distance_measure, output_path, search_k, max_dist)):
#
#         # noinspection PyBroadException
#         try:
#             mean_len_rank += len(ranking)
#
#             ranking['true_title'] = entity_title
#             hits = ranking.loc[ranking.guessed_title == entity_title].copy()
#
#             if len(hits) > 0:
#                 hits['success'] = True
#                 result = hits
#                 total_successes += 1
#                 mean_rank += result['rank'].min()
#             else:
#                 result = ranking.iloc[[0]].copy()
#                 result['success'] = False
#
#             results.append(result)
#
#             if len(results) >= save_interval:
#                 write_results()
#
#             data_sequence.\
#                 set_description('Total processed: {:.3f}. Success rate: {:.3f}. Mean rank: {:.3f}. '
#                                 'Mean len rank: {:.3f}.'. format(total_processed, total_successes / total_processed,
#                                                                  mean_rank / (total_successes + 1e-15),
#                                                                  mean_len_rank / total_processed))
#             if total_processed > max_iter:
#                 break
#
#         except:
#             print("Error: ", ranking, 'page_tile: ', entity_title)
#             # raise
#
#     write_results()
#
#     return result_path


# @click.command()
# @click.argument('all-entities-file', type=click.Path(exists=True), required=True, nargs=1)
# @click.argument('tagged_parquet', type=click.Path(exists=True), required=True, nargs=1)
# @click.argument('embedding_type', type=click.Choice(['flair']), required=True, nargs=1)
# @click.argument('ent_type', type=str, required=True, nargs=1)
# @click.argument('output_path', type=click.Path(exists=True), required=True, nargs=1)
# @click.option('--max-iter', type=float, default=np.inf)
# @click.option('--processes', type=int, default=6)
# @click.option('--w-size', type=int, default=10)
# @click.option('--batch-size', type=int, default=100)
# @click.option('--start-iteration', type=int, default=100)
# def build_context_matrix(all_entities_file, tagged_parquet, embedding_type, ent_type, output_path,
#                          processes=6, save_interval=100000, max_iter=np.inf, w_size=10, batch_size=100,
#                          start_iteration=0):
#
#     embeddings = load_embeddings(embedding_type)
#
#     print("Reading entity linking ground-truth file: {}.".format(tagged_parquet))
#     df = dd.read_parquet(tagged_parquet)
#     print("done.")
#
#     data_sequence = tqdm(df.iterrows(), total=len(df))
#
#     result_file = '{}/context-embeddings-embt_{}-entt_{}-wsize_{}.pkl'.\
#         format(output_path, embedding_type, ent_type, w_size)
#
#     all_entities = pd.read_pickle(all_entities_file)
#
#     all_entities = all_entities.loc[all_entities.TYPE == ent_type]
#
#     all_entities = all_entities.reset_index().reset_index().set_index('page_title').sort_index()
#
#     context_emb = None  # lazy creation
#
#     for it, link_result in \
#             enumerate(
#                 EmbedWithContext.run(embeddings, data_sequence, ent_type, w_size, batch_size,
#                                      processes, start_iteration=start_iteration)):
#         try:
#             if context_emb is None:
#                 dims = len(link_result.drop(['entity_title', 'count']).astype(np.float32).values)
#                 context_emb = np.zeros([len(all_entities), dims + 1], dtype=np.float32)
#
#             if it % save_interval == 0:
#                 print('Saving ...')
#
#                 pd.DataFrame(context_emb, index=all_entities.index).to_pickle(result_file)
#
#             idx = all_entities.loc[link_result.entity_title]['index']
#
#             context_emb[idx, 1:] += link_result.drop(['entity_title', 'count']).astype(np.float32).values
#
#             context_emb[idx, 0] += float(link_result['count'])
#
#             data_sequence.set_description('#entity links processed: {}'.format(it))
#
#         except:
#             print("Error: ", link_result)
#             raise
#
#         if it >= max_iter:
#             break
#
#     pd.DataFrame(context_emb, index=all_entities.index).to_pickle(result_file)
#
#     return result_file


@click.command()
@click.argument('context-matrix-file', type=click.Path(exists=True), required=True, nargs=1)
@click.argument('n-trees', type=int, required=True, nargs=1)
@click.argument('distance-measure', type=click.Choice(['angular', 'euclidean']), required=True, nargs=1)
def build_from_context_matrix(context_matrix_file, n_trees, distance_measure):

    build_from_matrix(context_matrix_file, distance_measure, n_trees)


def links_per_entity(context_matrix_file):

    df = pd.read_pickle(context_matrix_file)

    # Approximate number of links per entity:
    return (df.iloc[:, 0]/df.index.str.split('_').str.len().values).sort_values(ascending=False)


# @click.command()
# @click.argument('index-file', type=click.Path(exists=True), required=True, nargs=1)
# @click.argument('mapping-file', type=click.Path(exists=True), required=True, nargs=1)
# @click.argument('tagged_parquet', type=click.Path(exists=True), required=True, nargs=1)
# @click.argument('embedding_type', type=click.Choice(['flair']), required=True, nargs=1)
# @click.argument('ent_type', type=str, required=True, nargs=1)
# @click.argument('distance-measure', type=click.Choice(['angular', 'euclidean']), required=True, nargs=1)
# @click.argument('output_path', type=click.Path(exists=True), required=True, nargs=1)
# @click.option('--max-iter', type=float, default=np.inf)
# @click.option('--processes', type=int, default=6)
# @click.option('--w-size', type=int, default=10)
# @click.option('--batch-size', type=int, default=100)
# @click.option('--start-iteration', type=int, default=100)
# @click.option('--search-k', type=int, default=500)
# def evaluate_with_context(index_file, mapping_file, tagged_parquet, embedding_type, ent_type, distance_measure,
#                           output_path, processes=6, save_interval=10000, max_iter=np.inf, w_size=10, batch_size=100,
#                           start_iteration=0, search_k=10):
#
#     embeddings = load_embeddings(embedding_type)
#
#     print("Reading entity linking ground-truth file: {}.".format(tagged_parquet))
#     df = dd.read_parquet(tagged_parquet)
#     print("done.")
#
#     data_sequence = tqdm(df.iterrows(), total=len(df))
#
#     result_path = '{}/nedstat-index_{}-sk_{}.parquet'.format(output_path, Path(index_file).stem, search_k)
#
#     print("Write result statistics to: {} .".format(result_path))
#
#     results = []
#
#     def write_results():
#
#         nonlocal results
#
#         if len(results) == 0:
#             return
#
#         res = pd.concat(results)
#
#         # noinspection PyArgumentList
#         table = pa.Table.from_pandas(res)
#
#         pq.write_to_dataset(table, root_path=result_path)
#
#         results = []
#
#     total_successes = mean_rank = 0
#
#     # The evaluation Semaphore makes sure that the LookUpBySurfaceAndContext task creation will not run away from
#     # the actual processing of those tasks. If that would happen it would result in ever increasing memory consumption.
#     evaluation_semaphore = Semaphore(batch_size)
#
#     for total_processed, (entity_title, result) in \
#             enumerate(
#                 LookUpBySurfaceAndContext.run(index_file, mapping_file, distance_measure, search_k, embeddings,
#                                               data_sequence, start_iteration, ent_type, w_size, batch_size, processes,
#                                               evaluation_semaphore)):
#         try:
#             result['true_title'] = entity_title
#
#             # noinspection PyUnresolvedReferences
#             if (result.guessed_title == entity_title).sum() > 0:
#
#                 result['success'] = True
#
#                 total_successes += 1
#                 mean_rank += result['rank'].iloc[0]
#             else:
#                 result['success'] = False
#
#             data_sequence. \
#                 set_description('Total processed: {:.3f}. Success rate: {:.3f} Mean rank: {:.3f}'.
#                                 format(total_processed, total_successes / (total_processed+1e-15),
#                                        mean_rank / (total_successes + 1e-15)))
#
#             evaluation_semaphore.release()
#
#             results.append(result)
#
#             if len(results) >= save_interval:
#                 write_results()
#
#             if total_processed >= max_iter:
#                 break
#
#         except:
#             print("Error: ", result)
#             raise
#
#     write_results()


# class RefineLookup:
#
#     cm = None
#
#     def __init__(self, entity_title, ranking, link_embedding):
#
#         self._entity_title = entity_title
#         self._ranking = ranking
#         self._link_embedding = link_embedding
#
#     def __call__(self, *args, **kwargs):
#
#         if len(self._ranking) == 1:
#             return self._entity_title, self._ranking
#
#         e = self._link_embedding.drop(['entity_title', 'count']).astype(np.float32).values
#         e /= float(self._link_embedding['count'])
#         e /= norm(e)
#
#         # noinspection PyBroadException
#         try:
#             order = np.argsort(np.square(
#                 RefineLookup.cm.loc[self._ranking.guessed_title].values - repmat(e, len(self._ranking), 1)
#                 ).sum(axis=1))
#         except:
#             import ipdb;ipdb.set_trace()
#             raise
#
#         self._ranking = \
#             self._ranking.iloc[order].\
#                 reset_index(drop=True).\
#                 drop(columns=['rank']).\
#                 reset_index().\
#                 rename(columns={'index': 'rank'})
#
#         return self._entity_title, self._ranking
#
#     @staticmethod
#     def _get_all(entities_file, data_sequence_1, data_sequence_2, embeddings_1, ent_type_1, split_parts, n_trees, distance_measure_1,
#                  output_path,
#                  search_k_1, max_dist, lookup_semaphore,
#                  embeddings_2, ent_type_2, w_size, batch_size, embed_semaphore, processes):
#
#         for total_processed, ((entity_title, ranking), link_embedding) in \
#                 enumerate(zip(
#                     LookUpBySurface.run(entities_file, {ent_type_1: embeddings_1}, data_sequence_1, split_parts,
#                                         processes, n_trees, distance_measure_1, output_path, search_k_1, max_dist,
#                                         sem=lookup_semaphore),
#                     EmbedWithContext.run(embeddings_2, data_sequence_2, ent_type_2, w_size, batch_size,
#                                          processes, sem=embed_semaphore))):
#
#             yield RefineLookup(entity_title, ranking, link_embedding)
#
#     @staticmethod
#     def run(entities_file, context_matrix_file, data_sequence_1, data_sequence_2, embeddings_1, ent_type_1, split_parts,
#             n_trees, distance_measure_1, output_path, search_k_1, max_dist, lookup_semaphore,
#             embeddings_2, ent_type_2, w_size, batch_size, embed_semaphore, processes,
#             refine_processes=0):
#
#         return \
#             prun(RefineLookup._get_all(entities_file, data_sequence_1, data_sequence_2, embeddings_1, ent_type_1,
#                                        split_parts, n_trees, distance_measure_1, output_path, search_k_1, max_dist,
#                                        lookup_semaphore,
#                                        embeddings_2, ent_type_2, w_size, batch_size, embed_semaphore, processes),
#                  initializer=RefineLookup.initialize, initargs=(context_matrix_file,), processes=refine_processes)
#
#     @staticmethod
#     def initialize(context_matrix_file):
#
#         cm = pd.read_pickle(context_matrix_file)
#         for idx in tqdm(range(len(cm))):
#
#             if cm.iloc[idx, 0] == 0:
#                 continue
#
#             cm.iloc[idx, 1:] = cm.iloc[idx, 1:] / norm(cm.iloc[idx, 1:])
#
#         cm = cm.iloc[:, 1:]
#
#         RefineLookup.cm = cm


# @click.command()
# @click.argument('entities-file', type=click.Path(exists=True), required=True, nargs=1)
# @click.argument('tagged-parquet', type=click.Path(exists=True), required=True, nargs=1)
# @click.argument('ent-type', type=str, required=True, nargs=1)
# @click.argument('embedding-type-1', type=click.Choice(['fasttext']), required=True, nargs=1)
# @click.argument('n-trees', type=int, required=True, nargs=1)
# @click.argument('distance-measure-1', type=click.Choice(['angular', 'euclidean']), required=True, nargs=1)
# @click.argument('embedding-type-2', type=click.Choice(['flair']), required=True, nargs=1)
# @click.argument('w-size', type=int, required=True, nargs=1)
# @click.argument('batch-size', type=int, required=True, nargs=1)
# @click.argument('output-path', type=click.Path(exists=True), required=True, nargs=1)
# @click.option('--search-k-1', type=int, default=50)
# @click.option('--max-dist', type=float, default=0.25)
# @click.option('--processes', type=int, default=6)
# @click.option('--save-interval', type=int, default=10000)
# @click.option('--max-iter', type=float, default=np.inf)
# def evaluate_combined(entities_file, tagged_parquet, ent_type,
#                       embedding_type_1, n_trees, distance_measure_1,
#                       embedding_type_2, w_size, batch_size,
#                       output_path,
#                       search_k_1=50, max_dist=0.25, processes=6, save_interval=10000,
#                       max_iter=np.inf, split_parts=True):
#
#     embeddings_1 = load_embeddings(embedding_type_1)
#
#     embeddings_2 = load_embeddings(embedding_type_2)
#
#     print("Reading entity linking ground-truth file: {}".format(tagged_parquet))
#     df = dd.read_parquet(tagged_parquet)
#     print("done.")
#
#     data_sequence_1 = tqdm(df.iterrows(), total=len(df))
#     data_sequence_2 = df.iterrows()
#
#     result_path = '{}/nedstat-embt1_{}-embt2_{}-entt_{}-nt_{}-dm1_{}-sk_{}-md_{}-wsize_{}.parquet'.\
#         format(output_path, embedding_type_1, embedding_type_2, ent_type, n_trees, distance_measure_1,
#                search_k_1, max_dist, w_size)
#
#     print("Write result statistics to: {} .".format(result_path))
#
#     total_successes = mean_rank = mean_len_rank = 0
#
#     results = []
#
#     context_matrix_file = '{}/context-embeddings-embt_{}-entt_{}-wsize_{}.pkl'.\
#         format(output_path, embedding_type_2, ent_type, w_size)
#
#     def write_results():
#
#         nonlocal results
#
#         if len(results) == 0:
#             return
#
#         res = pd.concat(results, sort=True)
#
#         # noinspection PyArgumentList
#         table = pa.Table.from_pandas(res)
#
#         pq.write_to_dataset(table, root_path=result_path)
#
#         results = []
#
#     # The Semaphores make sure that the task creation will not run away from each other.
#     # If that would happen it would result in ever increasing memory consumption.
#     lookup_semaphore = Semaphore(batch_size*2)
#     embed_semaphore = Semaphore(batch_size*2)
#
#     for total_processed, (entity_title, ranking) in \
#         enumerate(RefineLookup.run(entities_file, context_matrix_file, data_sequence_1, data_sequence_2, embeddings_1,
#                                    ent_type, split_parts, n_trees, distance_measure_1, output_path, search_k_1,
#                                    max_dist, lookup_semaphore, embeddings_2, ent_type, w_size, batch_size, embed_semaphore,
#                                    processes)):
#
#         # noinspection PyBroadException
#         try:
#             mean_len_rank += len(ranking)
#
#             ranking['true_title'] = entity_title
#             hits = ranking.loc[ranking.guessed_title == entity_title].copy()
#
#             if len(hits) > 0:
#                 hits['success'] = True
#                 result = hits
#                 total_successes += 1
#                 mean_rank += result['rank'].min()
#             else:
#                 result = ranking.iloc[[0]].copy()
#                 result['success'] = False
#
#             results.append(result)
#
#             if len(results) >= save_interval:
#                 write_results()
#
#             data_sequence_1.\
#                 set_description('Total processed: {:.3f}. Success rate: {:.3f}. Mean rank: {:.3f}. '
#                                 'Mean len rank: {:.3f}.'. format(total_processed, total_successes /
#                                                                  (total_processed + 1e-15),
#                                                                  mean_rank / (total_successes + 1e-15),
#                                                                  mean_len_rank / (total_processed + 1e-15)))
#             if total_processed > max_iter:
#                 break
#
#         except:
#             print("Error: ", ranking, 'page_tile: ', entity_title)
#             raise
#
#         lookup_semaphore.release()
#
#         if total_processed % batch_size == 0:
#             embed_semaphore.release()
#
#     write_results()
#
#     return result_path


class NEDDataTask:

    def __init__(self, page_id, text, tags, link_titles, page_title, ent_types=None):

        self._page_id = page_id
        self._text = text
        self._tags = tags
        self._link_titles = link_titles
        self._page_title = page_title

        if ent_types is None:
            self._ent_types = {'PER', 'LOC', 'ORG'}

    def __call__(self, *args, **kwargs):

        sentences = json.loads(self._text)
        link_titles = json.loads(self._link_titles)
        tags = json.loads(self._tags)

        df_sentence = []
        df_linking = []

        for sen, sen_link_titles, sen_tags in zip(sentences, link_titles, tags):

            if len(self._ent_types.intersection({t if len(t) < 3 else t[2:] for t in sen_tags})) == 0:
                # Do not further process sentences that do not contain a relevant linked entity of type "ent_types".
                continue

            tmp1 = {'id': [len(df_sentence)],
                    'text': [json.dumps(sen)],
                    'tags': [json.dumps(sen_tags)],
                    'entities': [json.dumps(sen_link_titles)],
                    'page_id': [self._page_id],
                    'page_title': [self._page_title]}

            sen_link_titles = [t for t in list(set(sen_link_titles)) if len(t) > 0]

            tmp2 = {'target': sen_link_titles, 'sentence': len(sen_link_titles) * [len(df_sentence)]}

            df_sentence.append(pd.DataFrame.from_dict(tmp1).reset_index(drop=True))
            df_linking.append(pd.DataFrame.from_dict(tmp2).reset_index(drop=True))

        if len(df_sentence) > 0:
            return pd.concat(df_sentence), pd.concat(df_linking)
        else:
            return None, None

    @staticmethod
    def get_all(tagged_sqlite_file, sem=None):

        with sqlite3.connect(tagged_sqlite_file) as read_conn:
            read_conn.execute('pragma journal_mode=wal')

            total = int(read_conn.execute('select count(*) from tagged;').fetchone()[0])

            pos = read_conn.cursor().execute('SELECT page_id, text, tags, link_titles, page_title from tagged')

            for page_id, text, tags, link_titles, page_title in tqdm(pos, total=total):

                if sem is not None:
                    while not sem.acquire(timeout=100):
                        pass

                yield NEDDataTask(page_id, text, tags, link_titles, page_title)


@click.command()
@click.argument('tagged-sqlite-file', type=click.Path(exists=True), required=True, nargs=1)
@click.argument('ned-sqlite-file', type=click.Path(exists=False), required=True, nargs=1)
@click.option('--processes', type=int, default=6, help="number of parallel processes. default: 6")
@click.option('--writequeue', type=int, default=1000, help="size of database write queue. default: 1000.")
def ned_sentence_data(tagged_sqlite_file, ned_sqlite_file, processes, writequeue):
    """

    TAGGED_SQLITE_FILE: A sqlite database file that contains all wikipedia articles where the relevant
    entities have been tagged. This is a database that gives per article access to the tagged sentences,
    it can be created using 'tag-wiki-entities2sqlite'.


    NED_SQLITE_FILE: Output database. This database gives fast per entity and per sentence access, i.e., it
    provides a fast answer to the question: "Give me all sentences where entity X is discussed."

    """

    first_write = True

    sentence_counter = 0
    link_counter = 0

    # prevent infinite growth of multiprocessing queue
    sem = Semaphore(writequeue)

    with sqlite3.connect(ned_sqlite_file) as write_conn:

        write_conn.execute('pragma journal_mode=wal')

        for df_sentence, df_linking in prun(NEDDataTask.get_all(tagged_sqlite_file, sem=sem), processes=processes):

            if df_sentence is None:
                sem.release()
                continue

            df_sentence['id'] += sentence_counter
            df_linking['sentence'] += sentence_counter
            df_linking['id'] = [link_counter + i for i in range(len(df_linking))]

            sentence_counter += len(df_sentence)
            link_counter += len(df_linking)

            df_sentence.set_index('id').to_sql('sentences', con=write_conn, if_exists='append', index_label='id')
            df_linking.set_index('id').to_sql('links', con=write_conn, if_exists='append', index_label='id')

            if first_write:
                write_conn.execute('create index idx_target on links(target);')
                write_conn.execute('create index idx_sentence on links(sentence);')
                write_conn.execute('create index idx_page_id on sentences(page_id);')
                write_conn.execute('create index idx_page_title on sentences(page_title);')

                first_write = False

            sem.release()


@click.command()
@click.argument('pairing-sql-file', type=click.Path(exists=False), required=True, nargs=1)
@click.argument('ned-sql-file', type=click.Path(exists=True), required=True, nargs=1)
@click.argument('entities-file', type=click.Path(exists=True), required=True, nargs=1)
@click.argument('embedding-type', type=click.Choice(['fasttext', 'bert']), required=True, nargs=1)
@click.argument('n-trees', type=int, required=True, nargs=1)
@click.argument('distance-measure', type=click.Choice(['angular', 'euclidean']), required=True, nargs=1)
@click.argument('entity-index-path', type=click.Path(exists=True), required=True, nargs=1)
@click.option('--subset-file', type=click.Path(exists=True), default=None)
@click.option('--nsamples', type=int, default=None)
@click.option('--bad-count', type=int, default=10)
@click.option('--lookup-processes', type=int, default=2)
@click.option('--pairing-processes', type=int, default=10)
@click.option('--embedding-model', type=click.Path(exists=True), default=None)
@click.option('--layers', type=str, default="-1,-2,-3,-4", help="Which layers to use. default -1,-2,-3,-4")
@click.option('--pooling', type=str, default="first", help="How to pool the output for different tokens/words. "
                                                           "default: first.")
@click.option('--scalar-mix', type=bool, is_flag=True, help="Use scalar mix of layers.")
def ned_pairing(pairing_sql_file, ned_sql_file, entities_file, embedding_type, n_trees, distance_measure,
                entity_index_path, subset_file, nsamples, bad_count, lookup_processes, pairing_processes,
                embedding_model, layers, pooling, scalar_mix, search_k=50, max_dist=0.25):

    if nsamples is None:
        nsamples = np.inf

    epoch_size = 1000
    max_seq_length = 128
    tokenizer = None
    sen_subset = None

    if subset_file is not None:
        sen_subset = pd.read_pickle(subset_file)

    # embs = load_embeddings(embedding_type, model_path=embedding_model)
    embs = load_embeddings(embedding_type, model_path=embedding_model, layers=layers, pooling_operation=pooling,
                           use_scalar_mix=scalar_mix)

    embeddings = {'PER': embs, 'LOC': embs, 'ORG': embs}

    with sqlite3.connect(pairing_sql_file) as write_conn:

        write_conn.execute('pragma journal_mode=wal')

        for count, (id_a, id_b, sen_a, sen_b, pos_a, pos_b, end_a, end_b, label) in\
            enumerate(tqdm(WikipediaDataset(epoch_size, max_seq_length, tokenizer, ned_sql_file, entities_file,
                                            embeddings, n_trees, distance_measure, entity_index_path, search_k,
                                            max_dist, sen_subset, bad_count, lookup_processes,
                                            pairing_processes).get_sentence_pairs(), total=nsamples)):

            df = pd.DataFrame.from_dict({'id': count, 'id_a': [json.dumps(id_a)], 'id_b': [json.dumps(id_b)],
                                         'sen_a': [json.dumps(sen_a)], 'sen_b': [json.dumps(sen_b)],
                                         'pos_a': [pos_a], 'pos_b': [pos_b], 'end_a': [end_a], 'end_b': [end_b],
                                         'label': [label]}).set_index('id')

            df.to_sql('pairs', con=write_conn, if_exists='append', index_label='id')

            if float(count) > nsamples:
                break


@click.command()
@click.argument('ned-sql-file', type=click.Path(exists=True), required=True, nargs=1)
@click.argument('train-set-file', type=click.Path(exists=False), required=True, nargs=1)
@click.argument('test-set-file', type=click.Path(exists=False), required=True, nargs=1)
@click.option('--fraction-train', type=float, default=0.5, help="fraction of training data.")
def ned_train_test_split(ned_sql_file, train_set_file, test_set_file, fraction_train):
    """
    Splits the sentence data into train and test set.

    NED_SQL_FILE: See ned-sentence-data.

    Output:

    TRAIN_SET_FILE: Pickled pandas DataFrame that contains the sentence ids of the training set.

    TEST_SET_FILE: Pickled pandas DataFrame that contains the sentence ids of the test set.

    """
    with sqlite3.connect(ned_sql_file) as conn:

        df = shuffle(pd.read_sql("select sentences.id from sentences;", conn))

        df_train = df.iloc[0:int(fraction_train*len(df))]
        df_test = df.iloc[int(fraction_train * len(df))+1:]

        df_train.to_pickle(train_set_file)
        df_test.to_pickle(test_set_file)


@click.command()
@click.argument('pairing-sql-file', type=click.Path(exists=True), required=True, nargs=1)
@click.argument('model-dir', type=click.Path(exists=True), required=True, nargs=1)
@click.option('--max-seq-length', type=int, default=128)
@click.option('--do-lower-case', is_flag=True)
@click.option('--nsamples', type=int, default=None)
def ned_pairing_examples(pairing_sql_file, model_dir, max_seq_length, do_lower_case=False, nsamples=None):

    if nsamples is None:
        nsamples = np.inf

    tokenizer = BertTokenizer.from_pretrained(model_dir, do_lower_case=do_lower_case)

    with sqlite3.connect(pairing_sql_file) as read_conn:

        read_conn.execute('pragma journal_mode=wal')

        total = int(read_conn.execute('select count(*) from pairs;').fetchone()[0])

        pos = read_conn.cursor().execute('SELECT id_a, id_b, sen_a, sen_b, '
                                         'pos_a, pos_b, end_a, end_b, label from pairs')

        for counter, (id_a, id_b, sen_a, sen_b, pos_a, pos_b, end_a, end_b, label) in enumerate(tqdm(pos, total=total)):

            sample = InputExample(guid="%s-%s" % (pairing_sql_file, "{}-{}".format(id_a, id_b)),
                                  text_a=json.loads(sen_a), text_b=json.loads(sen_b),
                                  pos_a=pos_a, pos_b=pos_b, end_a=end_a, end_b=end_b, label=label)

            features = convert_examples_to_features(sample, max_seq_length, tokenizer, mark_entities=True)

            print('CLASS: {}, ID: {}'.format(label, sample.guid))

            text = ""
            for s, t in zip(features.tokens, features.segment_ids):

                if t == 2 and not s.startswith('##'):
                    s = '[ENT]' + s

                if s == '[SEP]':
                    s = '\n{}\n'.format(s)
                else:
                    s = " {}".format(s) if not s.startswith('##') else s[2:]

                text = text + s

            print(text, '\n\n\n')

            if counter > nsamples:
                break




























