from .index import build as build_index
from .index import load as load_index
from .embeddings.fasttext import FastTextEmbeddings
import click
import pandas as pd


@click.command()
@click.argument('all_entities_file', type=click.Path(exists=True), required=True, nargs=1)
@click.argument('embedding_type', type=click.Choice(['fasttext']), required=True, nargs=1)
@click.argument('ent_type', type=str, required=True, nargs=1)
@click.argument('n_trees', type=int, required=True, nargs=1)
@click.argument('n_processes', type=int, required=True, nargs=1)
@click.argument('distance_measure', type=click.Choice(['angular', 'euclidean']), required=True, nargs=1)
@click.argument('output_path', type=click.Path(exists=True), required=True, nargs=1)
def build(all_entities_file, embedding_type, ent_type, n_trees, n_processes, distance_measure, output_path):

    all_entities = pd.read_pickle(all_entities_file)

    embeddings = None
    if embedding_type == 'fasttext':
        embeddings = FastTextEmbeddings('../data/fasttext/cc.de.300.bin')

    build_index(all_entities, embeddings, ent_type, n_trees, n_processes, distance_measure, output_path)


def evaluate(all_entities_file, embedding_type, ent_type, n_trees,
             n_processes, distance_measure, output_path):

    all_entities = pd.read_pickle(all_entities_file)

    embeddings = None
    if embedding_type == 'fasttext':
        embeddings = FastTextEmbeddings('../data/fasttext/cc.de.300.bin')

    index, mapping = load_index(embeddings, ent_type, n_trees, distance_measure, output_path)
