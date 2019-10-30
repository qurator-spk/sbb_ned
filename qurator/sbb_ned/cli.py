from .index import build as build_index
from .index import build_from_matrix
from .index import load as load_index
from .index import best_matches
from .embeddings.base import load_embeddings, get_embedding_vectors, EmbedWithContext
import click
import numpy as np
import pandas as pd
import dask.dataframe as dd
import json
from tqdm import tqdm as tqdm
from qurator.utils.parallel import run as prun

import pyarrow as pa
import pyarrow.parquet as pq

from annoy import AnnoyIndex
from multiprocessing import Semaphore

from pathlib import Path


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


class EvalTask:

    index = None
    embeddings = None
    mapping = None
    search_k = None
    max_dist = None

    def __init__(self, page_title, entity_surface_parts, entity_title, split_parts):

        self._entity_surface_parts = entity_surface_parts
        self._entity_title = entity_title
        self._page_title = page_title
        self._split_parts = split_parts

    def __call__(self, *args, **kwargs):

        surface_text = " ".join(self._entity_surface_parts)

        text_embeddings = get_embedding_vectors(EvalTask.embeddings, surface_text, self._split_parts)

        ranking, hits = best_matches(text_embeddings, EvalTask.index, EvalTask.mapping, EvalTask.search_k,
                                     EvalTask.max_dist)

        ranking['on_page'] = self._page_title
        ranking['surface'] = surface_text

        return self._entity_title, ranking

    @staticmethod
    def _get_all(data_sequence, ent_type, split_parts):

        for _, article in data_sequence:

            sentences = json.loads(article.text)
            sen_link_titles = json.loads(article.link_titles)
            sen_tags = json.loads(article.tags)

            if ent_type not in set([t if len(t) < 3 else t[2:] for tags in sen_tags for t in tags]):
                # Do not further process articles that do not have any linked relevant entity of type "ent_type".
                continue

            for sen, link_titles, tags in zip(sentences, sen_link_titles, sen_tags):

                if ent_type not in set([t if len(t) < 3 else t[2:] for t in tags]):
                    # Do not further process sentences that do not contain a relevant linked entity of type "ent_type".
                    continue

                entity_surface_parts = []
                entity_title = ''
                for word, link_title, tag in zip(sen, link_titles, tags):

                    if (tag == 'O' or tag.startswith('B-')) and len(entity_surface_parts) > 0:

                        yield EvalTask(article.page_title, entity_surface_parts, entity_title, split_parts)
                        entity_surface_parts = []

                    if tag != 'O' and tag[2:] == ent_type:

                        entity_surface_parts.append(word)
                        entity_title = link_title

                if len(entity_surface_parts) > 0:
                    yield EvalTask(article.page_title, entity_surface_parts, entity_title, split_parts)

    @staticmethod
    def run(embeddings, data_sequence, ent_type, split_parts, processes, n_trees, distance_measure, output_path,
            search_k, max_dist):

        return prun(EvalTask._get_all(data_sequence, ent_type, split_parts), processes=processes,
                    initializer=EvalTask.initialize,
                    initargs=(embeddings, ent_type, n_trees, distance_measure, output_path, search_k, max_dist))

    @staticmethod
    def initialize(embeddings, ent_type, n_trees, distance_measure, output_path, search_k, max_dist):

        if type(embeddings) == tuple:
            EvalTask.embeddings = embeddings[0](**embeddings[1])
        else:
            EvalTask.embeddings = embeddings

        EvalTask.index, EvalTask.mapping = load_index(EvalTask.embeddings.config(), ent_type, n_trees,
                                                      distance_measure, output_path)

        EvalTask.search_k = search_k
        EvalTask.max_dist = max_dist


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

    total_processed = total_successes = mean_rank = mean_len_rank = 0

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

    for entity_title, ranking in EvalTask.run(embeddings, data_sequence, ent_type, split_parts, processes, n_trees,
                                              distance_measure, output_path, search_k, max_dist):

        # noinspection PyBroadException
        try:
            total_processed += 1

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

    it = 0
    for result in EmbedWithContext.run(embeddings, data_sequence, start_iteration, ent_type, w_size, batch_size,
                                       processes):
        try:
            it += len(result)

            if it % save_interval == 0:
                print('Saving ...')

                pd.DataFrame(context_emb, index=all_entities.index).to_pickle(result_file)

            for _, link_result in result.iterrows():

                idx = all_entities.loc[link_result.entity_title]['index']

                context_emb[idx, 1:] += link_result.drop(['entity_title', 'count']).astype(np.float32).values

                context_emb[idx, 0] += float(link_result['count'])

            data_sequence.set_description('#entity links processed: {}'.format(it))

        except:
            print("Error: ", result)
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


class EvalWithContextTask:

    index = None
    mapping = None
    search_k = None

    def __init__(self, link_result):

        self._link_result = link_result

    def __call__(self, *args, **kwargs):

        e = self._link_result.drop(['entity_title', 'count']).astype(np.float32).values
        e /= float(self._link_result['count'])

        ann_indices, dist = EvalWithContextTask.index.get_nns_by_vector(e, EvalWithContextTask.search_k,
                                                                        include_distances=True)

        hits = EvalWithContextTask.mapping.loc[ann_indices]
        hits['dist'] = dist

        hits = hits.sort_values('dist', ascending=True).reset_index(drop=True).reset_index(). \
            rename(columns={'index': 'rank', 'page_title': 'guessed_title'})

        success = hits.loc[hits.guessed_title == self._link_result.entity_title]

        if len(success) > 0:
            return self._link_result.entity_title, success
        else:
            return self._link_result.entity_title, hits.iloc[[0]]

    @staticmethod
    def initialize(index_file, mapping_file, dims, distance_measure, search_k):

        EvalWithContextTask.search_k = search_k

        EvalWithContextTask.index = AnnoyIndex(dims, distance_measure)

        EvalWithContextTask.index.load(index_file)

        EvalWithContextTask.mapping = pd.read_pickle(mapping_file)

    @staticmethod
    def _get_all(embeddings, data_sequence, start_iteration, ent_type, w_size, batch_size, processes,
                 evalutation_semaphore=None):

        # The embed semaphore makes sure that the EmbedWithContext will not over produce results in relation
        # to the EvalWithContextTask creation
        embed_semaphore = Semaphore(100)

        for result in EmbedWithContext.run(embeddings, data_sequence, start_iteration, ent_type, w_size, batch_size,
                                           processes, embed_semaphore):
            try:
                for _, link_result in result.iterrows():

                    if evalutation_semaphore is not None:
                        evalutation_semaphore.acquire()

                    yield EvalWithContextTask(link_result)

                embed_semaphore.release()

            except:
                print("Error: ", result)
                raise

    @staticmethod
    def run(index_file, mapping_file, dims, distance_measure, search_k,
            embeddings, data_sequence, start_iteration, ent_type, w_size, batch_size, processes, sem=None):

        return prun(EvalWithContextTask._get_all(embeddings, data_sequence, start_iteration, ent_type, w_size,
                                                 batch_size, processes, sem), processes=3*processes,
                    initializer=EvalWithContextTask.initialize,
                    initargs=(index_file, mapping_file, dims, distance_measure, search_k))


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

    # The evaluation Semaphore makes sure that the EvalWithContext task creation will not run away from
    # the actual processing of those tasks. If that would happen it would result in ever increasing memory consumption.
    evaluation_semaphore = Semaphore(100)

    for total_processed, (entity_title, result) in \
            enumerate(
                EvalWithContextTask.run(index_file, mapping_file, dims, distance_measure, search_k,
                                        embeddings, data_sequence, start_iteration, ent_type, w_size, batch_size,
                                        processes, evaluation_semaphore)):
        try:

            result['true_title'] = entity_title

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

        except:
            print("Error: ", result)
            raise

        if total_processed >= max_iter:
            break
