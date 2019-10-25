from .index import build as build_index
from .index import load as load_index
from .index import best_matches
from .index import get_embedding_vectors
from .embeddings.fasttext import FastTextEmbeddings
from .embeddings.flair import FlairEmbeddings
import click
import numpy as np
import pandas as pd
import dask.dataframe as dd
import json
from tqdm import tqdm as tqdm
from qurator.utils.parallel import run as prun

import pyarrow as pa
import pyarrow.parquet as pq

import multiprocessing as mp


def load_embeddings(embedding_type):

    print("Loading embeddings ...")

    if embedding_type == 'fasttext':
        embeddings = FastTextEmbeddings('../data/fasttext/cc.de.300.bin')
        dims = FastTextEmbeddings.dims()
    elif embedding_type == 'flair':

        # flair uses torch and as a consequence CUDA
        # CUDA does not work with the standard multiprocessing fork method, therefore we have to switch to spawn.
        mp.set_start_method('spawn')

        embeddings = (FlairEmbeddings, {'forward': 'de-forward', 'backward': 'de-backward', 'use_tokenizer': False})
        dims = FlairEmbeddings.dims()
    else:
        raise RuntimeError('Unknown embedding type: {}'.format(embedding_type))

    print('done')

    return embeddings, dims


@click.command()
@click.argument('all-entities-file', type=click.Path(exists=True), required=True, nargs=1)
@click.argument('embedding-type', type=click.Choice(['fasttext', 'flair']), required=True, nargs=1)
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

    build_index(all_entities, embeddings, dims, entity_type, n_trees, n_processes, distance_measure,
                split_parts, output_path)


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

        return self._page_title, ranking, self._entity_title

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
@click.argument('tagged_parquet', type=click.Path(exists=True), required=True, nargs=1)
@click.argument('embedding_type', type=click.Choice(['fasttext']), required=True, nargs=1)
@click.argument('ent_type', type=str, required=True, nargs=1)
@click.argument('n_trees', type=int, required=True, nargs=1)
@click.argument('distance_measure', type=click.Choice(['angular', 'euclidean']), required=True, nargs=1)
@click.argument('output_path', type=click.Path(exists=True), required=True, nargs=1)
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

    def get_eval_tasks():

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

    for page_title, ranking, ent_title in \
            prun(get_eval_tasks(), processes=processes, initializer=EvalTask.initialize,
                 initargs=(embeddings, ent_type, n_trees, distance_measure, output_path, search_k, max_dist)):

        # noinspection PyBroadException
        try:
            total_processed += 1

            mean_len_rank += len(ranking)

            ranking['true_title'] = ent_title
            hits = ranking.loc[ranking.guessed_title == ent_title].copy()

            if len(hits) > 0:
                hits['success'] = True
                result = hits
                total_successes += 1
                mean_rank += result['rank'].min()
            else:
                result = ranking.iloc[[0]].copy()
                result['success'] = False

            results.append(result)

            if len(results) % save_interval == 0:
                write_results()

            data_sequence.\
                set_description('Total processed: {:.3f}. Success rate: {:.3f}. Mean rank: {:.3f}. '
                                'Mean len rank: {:.3f}.'. format(total_processed, total_successes / total_processed,
                                                                 mean_rank / (total_successes + 1e-15),
                                                                 mean_len_rank / total_processed))
            if total_processed > max_iter:
                break

        except:
            print("Error: ", ranking, 'page_tile: ', ent_title)
            # raise

    write_results()

    return result_path


class BuildTask:

    embeddings = None

    def __init__(self, batch):

        self._batch = batch

    def __call__(self, *args, **kwargs):

        entity_titles = [entity_title for entity_title, _, _ in self._batch]
        entity_positions = [positions for _, _, positions in self._batch]
        sentences = [" ".join(sentence) for _, sentence, _ in self._batch]

        entity_parts = []
        vectors = []
        entity_id = []
        for e_idx, entity_part, embedding in BuildTask.embeddings.get(sentences, entity_positions):

            entity_parts.append(entity_part)
            vectors.append(embedding.astype(np.float32))
            entity_id.append(e_idx)

        vectors = pd.DataFrame(vectors, index=entity_parts)
        vectors['entity_id'] = entity_id

        text_embeddings = []

        for entity_id, text_emb in vectors.groupby('entity_id'):

            try:
                assert len(self._batch[entity_id][2]) == len(text_emb)

            except AssertionError:
                print(sentences[entity_id])
                print(text_emb.index)
                raise

            tmp = text_emb.drop(columns=['entity_id']).mean() * len(text_emb)

            tmp['entity_title'] = entity_titles[entity_id]
            tmp['count'] = len(text_emb)

            text_embeddings.append(tmp)

        ret = pd.concat(text_embeddings, axis=1).T

        return ret

    @staticmethod
    def initialize(embeddings):

        if type(embeddings) == tuple:
            BuildTask.embeddings = embeddings[0](**embeddings[1])
        else:
            BuildTask.embeddings = embeddings


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
def build_with_context(all_entities_file, tagged_parquet, embedding_type, ent_type, output_path,
                       processes=6, save_interval=100000, max_iter=np.inf, w_size=10, batch_size=100):

    embeddings, dims = load_embeddings(embedding_type)

    print("Reading entity linking ground-truth file: {}.".format(tagged_parquet))
    df = dd.read_parquet(tagged_parquet)
    print("done.")

    data_sequence = tqdm(df.iterrows(), total=len(df))

    result_file = '{}/context-embeddings-embt_{}-entt_{}-wsize_{}.pkl'.\
        format(output_path, embedding_type, ent_type, w_size)

    def compute_window(sen, entity_positions):

        start_pos = max(min(entity_positions) - w_size, 0)
        end_pos = min(max(entity_positions) + w_size, len(sen))
        sen_part = sen[start_pos:end_pos]

        # Example 1: min(entity_positions) = 20, w_size = 10
        # rel_start = 10 + min(20 - 10, 0) = 10
        # Example 2: min(entity_positions) = 5, w_size = 10
        # rel_start = 10 + min(5  - 10, 0) = 5
        rel_start = w_size + min(min(entity_positions) - w_size, 0)

        rel_positions = [i for i in range(rel_start, rel_start + len(entity_positions))]

        return sen_part, rel_positions

    def get_build_tasks():

        batch = []

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

                entity_positions = []
                entity_title = ''
                for pos, (word, link_title, tag) in enumerate(zip(sen, link_titles, tags)):

                    if (tag == 'O' or tag.startswith('B-')) and len(entity_positions) > 0:

                        sen_part, rel_positions = compute_window(sen, entity_positions)

                        batch.append((entity_title, sen_part, rel_positions))

                        entity_positions = []

                        if len(batch) >= batch_size:
                            yield BuildTask(batch)
                            batch = []

                    if tag != 'O' and tag[2:] == ent_type:

                        entity_positions.append(pos)
                        entity_title = link_title

                if len(entity_positions) > 0:

                    sen_part, rel_positions = compute_window(sen, entity_positions)

                    batch.append((entity_title, sen_part, rel_positions))

                    if len(batch) >= batch_size:
                        yield BuildTask(batch)
                        batch = []

    all_entities = pd.read_pickle(all_entities_file)

    all_entities = all_entities.loc[all_entities.TYPE == ent_type]

    all_entities = all_entities.reset_index().reset_index().set_index('page_title').sort_index()

    context_emb = np.zeros([len(all_entities), dims + 1], dtype=np.float32)

    it = 0
    for result in prun(get_build_tasks(), processes=processes, initializer=BuildTask.initialize,
                       initargs=(embeddings,)):

        try:
            it += len(result)

            if it % save_interval == 0:
                print('Saving ...')

                pd.DataFrame(context_emb, index=all_entities.index).to_pickle(result_file)

            for _, embeddings in result.iterrows():

                idx = all_entities.loc[embeddings.entity_title]['index']

                context_emb[idx, 1:] += embeddings.drop(['entity_title', 'count']).astype(np.float32).values

                context_emb[idx, 0] += float(embeddings['count'])

            data_sequence.set_description('#entity links processed: {}'.format(it))

        except:
            print("Error: ", result)
            raise

        if it >= max_iter:
            break

    pd.DataFrame(context_emb, index=all_entities.index).to_pickle(result_file)

    return result_file


def optimize():
    pass