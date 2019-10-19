from .index import build as build_index
from .index import load as load_index
from .index import best_matches
from .index import get_embedding_vectors
from .embeddings.fasttext import FastTextEmbeddings
from .embeddings.flair import FlairEmbeddings
import click
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
    elif embedding_type == 'flair':

        # flair uses torch and as a consequence CUDA
        # CUDA does not work with the standard multiprocessing fork method, therefore we have to switch to spawn.
        mp.set_start_method('spawn')

        embeddings = FlairEmbeddings('de-forward', 'de-backward')
    else:
        raise RuntimeError('Unknown embedding type: {}'.format(embedding_type))

    print('done')

    return embeddings


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

    embeddings = load_embeddings(embedding_type)

    build_index(all_entities, embeddings, entity_type, n_trees, n_processes, distance_measure, split_parts, output_path)


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
def evaluate(tagged_parquet, embedding_type, ent_type, n_trees,
             distance_measure, output_path, search_k=50, max_dist=0.25, processes=6, save_interval=10000,
             split_parts=True, max_iter=10000):

    embeddings = load_embeddings(embedding_type)

    print("Reading entity linking ground-truth file: {}".format(tagged_parquet))
    df = dd.read_parquet(tagged_parquet)
    print("done.")

    progress = tqdm(df.iterrows(), total=len(df))

    result_path = 'nedstat_embt{}_entt{}_nt{}_dm{}_sk{}_md{}.parquet'.format(embedding_type, ent_type, n_trees,
                                                                             distance_measure, search_k, max_dist)
    print("Write result statistics to: {} .".format(result_path))

    def get_eval_tasks():

        for _, article in progress:

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

                    if tag == 'O' or tag.startswith('B-'):

                        if len(entity_surface_parts) > 0:
                            yield EvalTask(article.page_title, entity_surface_parts, entity_title, split_parts)
                            entity_surface_parts = []

                    elif tag != 'O':
                        if tag[2:] == ent_type:

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

            progress.set_description('Total processed: {:.3f}. Success rate: {:.3f}. Mean rank: {:.3f}. '
                                     'Mean len rank: {:.3f}.'.
                                     format(total_processed, total_successes / total_processed,
                                            mean_rank / (total_successes + 1e-15), mean_len_rank / total_processed))
            if total_processed > max_iter:
                break

        except:
            print("Error: ", ranking, 'page_tile: ', ent_title)
            raise

    write_results()

    return result_path


def optimize():
    pass