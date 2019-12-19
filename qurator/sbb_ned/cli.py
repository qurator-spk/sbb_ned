import warnings
import logging

warnings.filterwarnings('ignore', category=FutureWarning)

from .index import build as build_index
from .index import build_from_matrix, LookUpBySurface, LookUpBySurfaceAndContext
from .embeddings.base import load_embeddings, EmbedWithContext
from .ground_truth.data_processor import WikipediaDataset
import click
import numpy as np
import pandas as pd
import dask.dataframe as dd
from tqdm import tqdm as tqdm

import pyarrow as pa
import pyarrow.parquet as pq

from multiprocessing import Semaphore

from pathlib import Path
from qurator.utils.parallel import run as prun
from numpy.linalg import norm
from numpy.matlib import repmat

import json
import sqlite3

logger = logging.getLogger(__name__)


@click.command()
@click.argument('all-entities-file', type=click.Path(exists=True), required=True, nargs=1)
@click.argument('embedding-type', type=click.Choice(['fasttext']), required=True, nargs=1)
@click.argument('entity-type', type=str, required=True, nargs=1)
@click.argument('n-trees', type=int, required=True, nargs=1)
@click.argument('output-path', type=click.Path(exists=True), required=True, nargs=1)
@click.option('--n-processes', type=int, default=6)
@click.option('--distance-measure', type=click.Choice(['angular', 'euclidean']), default='angular')
@click.option('--split-parts', type=bool, default=True)
def build(all_entities_file, embedding_type, entity_type, n_trees, output_path,
          n_processes, distance_measure, split_parts):

    all_entities = pd.read_pickle(all_entities_file)

    embeddings, dims = load_embeddings(embedding_type)

    build_index(all_entities, embeddings, dims, entity_type, n_trees, n_processes, distance_measure, split_parts,
                output_path)


@click.command()
@click.argument('tagged-parquet', type=click.Path(exists=True), required=True, nargs=1)
@click.argument('embedding-type', type=click.Choice(['fasttext']), required=True, nargs=1)
@click.argument('ent-type', type=str, required=True, nargs=1)
@click.argument('n-trees', type=int, required=True, nargs=1)
@click.argument('distance-measure', type=click.Choice(['angular', 'euclidean']), required=True, nargs=1)
@click.argument('output-path', type=click.Path(exists=True), required=True, nargs=1)
@click.option('--max-iter', type=float, default=np.inf)
def evaluate(tagged_parquet, embedding_type, ent_type, n_trees,
             distance_measure, output_path, search_k=50, max_dist=0.25, processes=6, save_interval=10000,
             split_parts=True, max_iter=np.inf):

    embeddings, dims = load_embeddings(embedding_type)

    print("Reading entity linking ground-truth file: {}".format(tagged_parquet))
    df = dd.read_parquet(tagged_parquet)
    print("done.")

    data_sequence = tqdm(df.iterrows(), total=len(df))

    result_path = '{}/nedstat-embt_{}-entt_{}-nt_{}-dm_{}-sk_{}-md_{}.parquet'.\
        format(output_path, embedding_type, ent_type, n_trees, distance_measure, search_k, max_dist)

    print("Write result statistics to: {} .".format(result_path))

    total_successes = mean_rank = mean_len_rank = 0

    results = []

    def write_results():

        nonlocal results

        if len(results) == 0:
            return

        res = pd.concat(results)

        # noinspection PyArgumentList
        table = pa.Table.from_pandas(res)

        pq.write_to_dataset(table, root_path=result_path)

        results = []

    for total_processed, (entity_title, ranking) in \
            enumerate(LookUpBySurface.run({ent_type: embeddings}, data_sequence, split_parts, processes, n_trees,
                                          distance_measure, output_path, search_k, max_dist)):

        # noinspection PyBroadException
        try:
            mean_len_rank += len(ranking)

            ranking['true_title'] = entity_title
            hits = ranking.loc[ranking.guessed_title == entity_title].copy()

            if len(hits) > 0:
                hits['success'] = True
                result = hits
                total_successes += 1
                mean_rank += result['rank'].min()
            else:
                result = ranking.iloc[[0]].copy()
                result['success'] = False

            results.append(result)

            if len(results) >= save_interval:
                write_results()

            data_sequence.\
                set_description('Total processed: {:.3f}. Success rate: {:.3f}. Mean rank: {:.3f}. '
                                'Mean len rank: {:.3f}.'. format(total_processed, total_successes / total_processed,
                                                                 mean_rank / (total_successes + 1e-15),
                                                                 mean_len_rank / total_processed))
            if total_processed > max_iter:
                break

        except:
            print("Error: ", ranking, 'page_tile: ', entity_title)
            # raise

    write_results()

    return result_path


@click.command()
@click.argument('all-entities-file', type=click.Path(exists=True), required=True, nargs=1)
@click.argument('tagged_parquet', type=click.Path(exists=True), required=True, nargs=1)
@click.argument('embedding_type', type=click.Choice(['flair']), required=True, nargs=1)
@click.argument('ent_type', type=str, required=True, nargs=1)
@click.argument('output_path', type=click.Path(exists=True), required=True, nargs=1)
@click.option('--max-iter', type=float, default=np.inf)
@click.option('--processes', type=int, default=6)
@click.option('--w-size', type=int, default=10)
@click.option('--batch-size', type=int, default=100)
@click.option('--start-iteration', type=int, default=100)
def build_context_matrix(all_entities_file, tagged_parquet, embedding_type, ent_type, output_path,
                         processes=6, save_interval=100000, max_iter=np.inf, w_size=10, batch_size=100,
                         start_iteration=0):

    embeddings, dims = load_embeddings(embedding_type)

    print("Reading entity linking ground-truth file: {}.".format(tagged_parquet))
    df = dd.read_parquet(tagged_parquet)
    print("done.")

    data_sequence = tqdm(df.iterrows(), total=len(df))

    result_file = '{}/context-embeddings-embt_{}-entt_{}-wsize_{}.pkl'.\
        format(output_path, embedding_type, ent_type, w_size)

    all_entities = pd.read_pickle(all_entities_file)

    all_entities = all_entities.loc[all_entities.TYPE == ent_type]

    all_entities = all_entities.reset_index().reset_index().set_index('page_title').sort_index()

    context_emb = np.zeros([len(all_entities), dims + 1], dtype=np.float32)

    for it, link_result in \
            enumerate(
                EmbedWithContext.run(embeddings, data_sequence, ent_type, w_size, batch_size,
                                     processes, start_iteration=start_iteration)):
        try:
            if it % save_interval == 0:
                print('Saving ...')

                pd.DataFrame(context_emb, index=all_entities.index).to_pickle(result_file)

            idx = all_entities.loc[link_result.entity_title]['index']

            context_emb[idx, 1:] += link_result.drop(['entity_title', 'count']).astype(np.float32).values

            context_emb[idx, 0] += float(link_result['count'])

            data_sequence.set_description('#entity links processed: {}'.format(it))

        except:
            print("Error: ", link_result)
            raise

        if it >= max_iter:
            break

    pd.DataFrame(context_emb, index=all_entities.index).to_pickle(result_file)

    return result_file


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


@click.command()
@click.argument('index-file', type=click.Path(exists=True), required=True, nargs=1)
@click.argument('mapping-file', type=click.Path(exists=True), required=True, nargs=1)
@click.argument('tagged_parquet', type=click.Path(exists=True), required=True, nargs=1)
@click.argument('embedding_type', type=click.Choice(['flair']), required=True, nargs=1)
@click.argument('ent_type', type=str, required=True, nargs=1)
@click.argument('distance-measure', type=click.Choice(['angular', 'euclidean']), required=True, nargs=1)
@click.argument('output_path', type=click.Path(exists=True), required=True, nargs=1)
@click.option('--max-iter', type=float, default=np.inf)
@click.option('--processes', type=int, default=6)
@click.option('--w-size', type=int, default=10)
@click.option('--batch-size', type=int, default=100)
@click.option('--start-iteration', type=int, default=100)
@click.option('--search-k', type=int, default=500)
def evaluate_with_context(index_file, mapping_file, tagged_parquet, embedding_type, ent_type, distance_measure,
                          output_path, processes=6, save_interval=10000, max_iter=np.inf, w_size=10, batch_size=100,
                          start_iteration=0, search_k=10):

    embeddings, dims = load_embeddings(embedding_type)

    print("Reading entity linking ground-truth file: {}.".format(tagged_parquet))
    df = dd.read_parquet(tagged_parquet)
    print("done.")

    data_sequence = tqdm(df.iterrows(), total=len(df))

    result_path = '{}/nedstat-index_{}-sk_{}.parquet'.format(output_path, Path(index_file).stem, search_k)

    print("Write result statistics to: {} .".format(result_path))

    results = []

    def write_results():

        nonlocal results

        if len(results) == 0:
            return

        res = pd.concat(results)

        # noinspection PyArgumentList
        table = pa.Table.from_pandas(res)

        pq.write_to_dataset(table, root_path=result_path)

        results = []

    total_successes = mean_rank = 0

    # The evaluation Semaphore makes sure that the LookUpBySurfaceAndContext task creation will not run away from
    # the actual processing of those tasks. If that would happen it would result in ever increasing memory consumption.
    evaluation_semaphore = Semaphore(batch_size)

    for total_processed, (entity_title, result) in \
            enumerate(
                LookUpBySurfaceAndContext.run(index_file, mapping_file, dims, distance_measure, search_k, embeddings,
                                              data_sequence, start_iteration, ent_type, w_size, batch_size, processes,
                                              evaluation_semaphore)):
        try:
            result['true_title'] = entity_title

            # noinspection PyUnresolvedReferences
            if (result.guessed_title == entity_title).sum() > 0:

                result['success'] = True

                total_successes += 1
                mean_rank += result['rank'].iloc[0]
            else:
                result['success'] = False

            data_sequence. \
                set_description('Total processed: {:.3f}. Success rate: {:.3f} Mean rank: {:.3f}'.
                                format(total_processed, total_successes / (total_processed+1e-15),
                                       mean_rank / (total_successes + 1e-15)))

            evaluation_semaphore.release()

            results.append(result)

            if len(results) >= save_interval:
                write_results()

            if total_processed >= max_iter:
                break

        except:
            print("Error: ", result)
            raise

    write_results()


class RefineLookup:

    cm = None

    def __init__(self, entity_title, ranking, link_embedding):

        self._entity_title = entity_title
        self._ranking = ranking
        self._link_embedding = link_embedding

    def __call__(self, *args, **kwargs):

        if len(self._ranking) == 1:
            return self._entity_title, self._ranking

        e = self._link_embedding.drop(['entity_title', 'count']).astype(np.float32).values
        e /= float(self._link_embedding['count'])
        e /= norm(e)

        # noinspection PyBroadException
        try:
            order = np.argsort(np.square(
                RefineLookup.cm.loc[self._ranking.guessed_title].values - repmat(e, len(self._ranking), 1)
                ).sum(axis=1))
        except:
            import ipdb;ipdb.set_trace()
            raise

        self._ranking = \
            self._ranking.iloc[order].\
                reset_index(drop=True).\
                drop(columns=['rank']).\
                reset_index().\
                rename(columns={'index': 'rank'})

        return self._entity_title, self._ranking

    @staticmethod
    def _get_all(data_sequence_1, data_sequence_2, embeddings_1, ent_type_1, split_parts, n_trees, distance_measure_1,
                 output_path,
                 search_k_1, max_dist, lookup_semaphore,
                 embeddings_2, ent_type_2, w_size, batch_size, embed_semaphore, processes):

        for total_processed, ((entity_title, ranking), link_embedding) in \
                enumerate(zip(
                    LookUpBySurface.run({ent_type_1: embeddings_1}, data_sequence_1, split_parts, processes, n_trees,
                                        distance_measure_1, output_path, search_k_1, max_dist, sem=lookup_semaphore),
                    EmbedWithContext.run(embeddings_2, data_sequence_2, ent_type_2, w_size, batch_size,
                                         processes, sem=embed_semaphore))):

            yield RefineLookup(entity_title, ranking, link_embedding)

    @staticmethod
    def run(context_matrix_file, data_sequence_1, data_sequence_2, embeddings_1, ent_type_1, split_parts, n_trees,
            distance_measure_1, output_path, search_k_1, max_dist, lookup_semaphore,
            embeddings_2, ent_type_2, w_size, batch_size, embed_semaphore, processes,
            refine_processes=0):

        return \
            prun(RefineLookup._get_all(data_sequence_1, data_sequence_2, embeddings_1, ent_type_1, split_parts,
                                       n_trees, distance_measure_1, output_path, search_k_1, max_dist,
                                       lookup_semaphore,
                                       embeddings_2, ent_type_2, w_size, batch_size, embed_semaphore, processes),
                 initializer=RefineLookup.initialize, initargs=(context_matrix_file,), processes=refine_processes)

    @staticmethod
    def initialize(context_matrix_file):

        cm = pd.read_pickle(context_matrix_file)
        for idx in tqdm(range(len(cm))):

            if cm.iloc[idx, 0] == 0:
                continue

            cm.iloc[idx, 1:] = cm.iloc[idx, 1:] / norm(cm.iloc[idx, 1:])

        cm = cm.iloc[:, 1:]

        RefineLookup.cm = cm


@click.command()
@click.argument('tagged-parquet', type=click.Path(exists=True), required=True, nargs=1)
@click.argument('ent-type', type=str, required=True, nargs=1)
@click.argument('embedding-type-1', type=click.Choice(['fasttext']), required=True, nargs=1)
@click.argument('n-trees', type=int, required=True, nargs=1)
@click.argument('distance-measure-1', type=click.Choice(['angular', 'euclidean']), required=True, nargs=1)
@click.argument('embedding-type-2', type=click.Choice(['flair']), required=True, nargs=1)
@click.argument('w-size', type=int, required=True, nargs=1)
@click.argument('batch-size', type=int, required=True, nargs=1)
@click.argument('output-path', type=click.Path(exists=True), required=True, nargs=1)
@click.option('--search-k-1', type=int, default=50)
@click.option('--max-dist', type=float, default=0.25)
@click.option('--processes', type=int, default=6)
@click.option('--save-interval', type=int, default=10000)
@click.option('--max-iter', type=float, default=np.inf)
def evaluate_combined(tagged_parquet, ent_type,
                      embedding_type_1, n_trees, distance_measure_1,
                      embedding_type_2, w_size, batch_size,
                      output_path,
                      search_k_1=50, max_dist=0.25, processes=6, save_interval=10000,
                      max_iter=np.inf, split_parts=True):

    embeddings_1, dims_1 = load_embeddings(embedding_type_1)

    embeddings_2, dims_2 = load_embeddings(embedding_type_2)

    print("Reading entity linking ground-truth file: {}".format(tagged_parquet))
    df = dd.read_parquet(tagged_parquet)
    print("done.")

    data_sequence_1 = tqdm(df.iterrows(), total=len(df))
    data_sequence_2 = df.iterrows()

    result_path = '{}/nedstat-embt1_{}-embt2_{}-entt_{}-nt_{}-dm1_{}-sk_{}-md_{}-wsize_{}.parquet'.\
        format(output_path, embedding_type_1, embedding_type_2, ent_type, n_trees, distance_measure_1,
               search_k_1, max_dist, w_size)

    print("Write result statistics to: {} .".format(result_path))

    total_successes = mean_rank = mean_len_rank = 0

    results = []

    context_matrix_file = '{}/context-embeddings-embt_{}-entt_{}-wsize_{}.pkl'.\
        format(output_path, embedding_type_2, ent_type, w_size)

    def write_results():

        nonlocal results

        if len(results) == 0:
            return

        res = pd.concat(results, sort=True)

        # noinspection PyArgumentList
        table = pa.Table.from_pandas(res)

        pq.write_to_dataset(table, root_path=result_path)

        results = []

    # The Semaphores make sure that the task creation will not run away from each other.
    # If that would happen it would result in ever increasing memory consumption.
    lookup_semaphore = Semaphore(batch_size*2)
    embed_semaphore = Semaphore(batch_size*2)

    for total_processed, (entity_title, ranking) in \
        enumerate(RefineLookup.run(context_matrix_file, data_sequence_1, data_sequence_2, embeddings_1, ent_type,
                                   split_parts, n_trees, distance_measure_1, output_path, search_k_1, max_dist,
                                   lookup_semaphore, embeddings_2, ent_type, w_size, batch_size, embed_semaphore,
                                   processes)):

        # noinspection PyBroadException
        try:
            mean_len_rank += len(ranking)

            ranking['true_title'] = entity_title
            hits = ranking.loc[ranking.guessed_title == entity_title].copy()

            if len(hits) > 0:
                hits['success'] = True
                result = hits
                total_successes += 1
                mean_rank += result['rank'].min()
            else:
                result = ranking.iloc[[0]].copy()
                result['success'] = False

            results.append(result)

            if len(results) >= save_interval:
                write_results()

            data_sequence_1.\
                set_description('Total processed: {:.3f}. Success rate: {:.3f}. Mean rank: {:.3f}. '
                                'Mean len rank: {:.3f}.'. format(total_processed, total_successes /
                                                                 (total_processed + 1e-15),
                                                                 mean_rank / (total_successes + 1e-15),
                                                                 mean_len_rank / (total_processed + 1e-15)))
            if total_processed > max_iter:
                break

        except:
            print("Error: ", ranking, 'page_tile: ', entity_title)
            raise

        lookup_semaphore.release()

        if total_processed % batch_size == 0:
            embed_semaphore.release()

    write_results()

    return result_path


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
    def get_all(tagged_sqlite_file):

        with sqlite3.connect(tagged_sqlite_file) as read_conn:
            read_conn.execute('pragma journal_mode=wal')

            total = int(read_conn.execute('select count(*) from tagged;').fetchone()[0])

            pos = read_conn.cursor().execute('SELECT page_id, text, tags, link_titles, page_title from tagged')

            for page_id, text, tags, link_titles, page_title in tqdm(pos, total=total):

                yield NEDDataTask(page_id, text, tags, link_titles, page_title)


@click.command()
@click.argument('tagged-sqlite-file', type=click.Path(exists=True), required=True, nargs=1)
@click.argument('ned-sqlite-file', type=click.Path(exists=False), required=True, nargs=1)
@click.option('--processes', type=int, default=6)
def per_sentence_ned_data(tagged_sqlite_file, ned_sqlite_file, processes):

    with sqlite3.connect(ned_sqlite_file) as write_conn:

        write_conn.execute('pragma journal_mode=wal')

        sentence_counter = 0
        link_counter = 0

        for df_sentence, df_linking in prun(NEDDataTask.get_all(tagged_sqlite_file), processes=processes):

            if df_sentence is None:
                continue

            df_sentence['id'] += sentence_counter
            df_linking['sentence'] += sentence_counter
            df_linking['id'] = [link_counter + i for i in range(len(df_linking))]

            sentence_counter += len(df_sentence)
            link_counter += len(df_linking)

            df_sentence.set_index('id').to_sql('sentences', con=write_conn, if_exists='append', index_label='id')
            df_linking.set_index('id').to_sql('links', con=write_conn, if_exists='append', index_label='id')

        write_conn.execute('create index idx_target on links(target);')
        write_conn.execute('create index idx_sentence on links(sentence);')
        write_conn.execute('create index idx_page_id on sentences(page_id);')
        write_conn.execute('create index idx_page_title on sentences(page_title);')


@click.command()
@click.argument('train-set-sql-file', type=click.Path(exists=False), required=True, nargs=1)
@click.argument('ned-sql-file', type=click.Path(exists=True), required=True, nargs=1)
@click.argument('entities-file', type=click.Path(exists=True), required=True, nargs=1)
@click.argument('embedding-type', type=click.Choice(['fasttext']), required=True, nargs=1)
@click.argument('n-trees', type=int, required=True, nargs=1)
@click.argument('distance-measure', type=click.Choice(['angular', 'euclidean']), required=True, nargs=1)
@click.argument('entity-index-path', type=click.Path(exists=True), required=True, nargs=1)
def ned_training_data(train_set_sql_file,
                      ned_sql_file, entities_file, embedding_type, n_trees, distance_measure, entity_index_path,
                      search_k=50, max_dist=0.25):

    epoch_size = 1000
    label_map = None
    max_seq_length = 128
    tokenizer = None

    embs, dims = load_embeddings(embedding_type)

    embeddings = {'PER': embs, 'LOC': embs, 'ORG': embs}

    with sqlite3.connect(train_set_sql_file) as write_conn:

        write_conn.execute('pragma journal_mode=wal')

        count = 0

        for sen_a, sen_b, pos_a, pos_b, label in\
            WikipediaDataset(epoch_size, label_map, max_seq_length, tokenizer,
                             ned_sql_file, entities_file, embeddings, n_trees, distance_measure, entity_index_path,
                             search_k, max_dist).get_sentence_pairs():

            df = pd.DataFrame.from_dict({'id': count, 'sen_a': [json.dumps(sen_a)], 'sen_b': [json.dumps(sen_b)],
                                         'pos_a': [pos_a], 'pos_b': [pos_b], 'label': [label]}).set_index('id')

            df .to_sql('links', con=write_conn, if_exists='append', index_label='id')

            count += 1







































