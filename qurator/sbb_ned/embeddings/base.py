import numpy as np
import pandas as pd
import re
import multiprocessing as mp
import json
from tqdm import tqdm as tqdm
import torch

from qurator.utils.parallel import run as prun


class Embeddings:

    def __init__(self, *args, **kwargs):
        pass

    def get(self, key):
        raise NotImplementedError()

    def description(self):
        raise NotImplementedError()


def get_embedding_vectors(embeddings, surface, split_parts):

    parts = []
    if type(surface) == str:

        if split_parts:
            parts = [p for p in re.split(r'[ \-_]', surface)]
        else:
            parts = [surface]

    elif type(surface) == list:

        if split_parts:
            parts = [p for s in surface for p in re.split(r'[ \-_]', s)]
        else:
            parts = surface
    else:
        RuntimeError('Type of surface not supported.')

    if split_parts:
        parts = [re.sub(r'[\W_]+', '', p) for p in parts]
    
    parts = [p.lower() for p in parts if len(p) > 0]

    vectors = []
    vector_parts = []
    for vp, emb in embeddings.get(parts):

        vector_parts.append(vp)
        vectors.append(emb.astype(np.float32))

    ret = pd.DataFrame(vectors, index=vector_parts).drop_duplicates()

    return ret


def load_embeddings(embedding_type, layers="-1, -2, -3, -4", **kwargs):

    print("Loading embeddings ...")

    if embedding_type == 'fasttext':

        from .fasttext import FastTextEmbeddings

        embeddings = FastTextEmbeddings(**kwargs)
    elif embedding_type == 'flair':

        from .flair import FlairEmbeddings

        # flair uses torch and as a consequence CUDA
        # CUDA does not work with the standard multiprocessing fork method, therefore we have to switch to spawn.
        mp.set_start_method('spawn')

        embeddings = (FlairEmbeddings, {'forward': 'de-forward', 'backward': 'de-backward', 'use_tokenizer': True})
        #embeddings = (FlairEmbeddings, {'forward': 'de-historic-rw-forward', 'backward': 'de-historic-rw-backward', 'use_tokenizer': True})
    elif embedding_type == 'bert':

        from .bert import BertEmbeddings

        # bert uses torch and as a consequence CUDA
        # CUDA does not work with the standard multiprocessing fork method, therefore we have to switch to spawn.
        mp.set_start_method('spawn')

        kwargs = dict(kwargs, layers=layers)

        embeddings = (BertEmbeddings, kwargs)

    else:
        raise RuntimeError('Unknown embedding type: {}'.format(embedding_type))

    print('done')

    return embeddings


class EmbedTask:
    embeddings = None

    def __init__(self, page_title, entity_label, split_parts):
        self._page_title = page_title
        self._entity_label = entity_label
        self._split_parts = split_parts

    def __call__(self, *args, **kwargs):

        if type(self._entity_label) == list:

            emb = []
            for el in self._entity_label:
                emb.append(get_embedding_vectors(EmbedTask.embeddings, el, self._split_parts))

            if len(emb) > 0:
                try:
                    emb = pd.concat(emb).sort_index()
                except:
                    import ipdb;ipdb.set_trace()

                emb = emb[~emb.index.duplicated(keep='first')]

                return self._page_title, emb, EmbedTask.embeddings.config()
            else:
                return self._page_title, pd.DataFrame(), EmbedTask.embeddings.config()
        else:
            return self._page_title, get_embedding_vectors(EmbedTask.embeddings, self._entity_label,
                                                           self._split_parts), EmbedTask.embeddings.config()

    @staticmethod
    def initialize(embeddings):

        if type(embeddings) == tuple:
            EmbedTask.embeddings = embeddings[0](**embeddings[1])
        else:
            EmbedTask.embeddings = embeddings

    @staticmethod
    def _get_all(all_entities, split_parts):
        for i, (page_title, entity) in tqdm(enumerate(all_entities.iterrows()), total=len(all_entities)):

            if i % 1000 == 0:
                torch.cuda.empty_cache()
            
            yield EmbedTask(page_title, entity.label, split_parts)

    @staticmethod
    def run(embeddings, all_entities, split_parts, processes):

        return prun(EmbedTask._get_all(all_entities, split_parts), processes=processes,
                    initializer=EmbedTask.initialize, initargs=(embeddings,))


class EmbedWithContext:

    embeddings = None
    sem = None

    def __init__(self, batch):

        self._batch = batch

    def __call__(self, *args, **kwargs):

        entity_titles = [entity_title for entity_title, _, _ in self._batch]
        entity_positions = [positions for _, _, positions in self._batch]
        sentences = [" ".join(sentence) for _, sentence, _ in self._batch]

        entity_parts = []
        vectors = []
        entity_id = []
        for e_idx, entity_part, embedding in EmbedWithContext.embeddings.get(sentences, entity_positions):

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
            EmbedWithContext.embeddings = embeddings[0](**embeddings[1])
        else:
            EmbedWithContext.embeddings = embeddings

    @staticmethod
    def _get_all(data_sequence, start_iteration, ent_type, wnd_size, batch_size, embed_semaphore=None):

        def compute_window(sentence, positions):

            start_pos = max(min(positions) - wnd_size, 0)
            end_pos = min(max(positions) + wnd_size, len(sentence))
            sentence_part = sentence[start_pos:end_pos]

            # Example 1: min(entity_positions) = 20, w_size = 10
            # rel_start = 10 + min(20 - 10, 0) = 10
            # Example 2: min(entity_positions) = 5, w_size = 10
            # rel_start = 10 + min(5  - 10, 0) = 5
            rel_start = wnd_size + min(min(positions) - wnd_size, 0)

            wnd_positions = [i for i in range(rel_start, rel_start + len(positions))]

            return sentence_part, wnd_positions

        batch = []

        for a_num, (_, article) in enumerate(data_sequence):

            if a_num < start_iteration:
                continue

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

                            if embed_semaphore is not None:
                                embed_semaphore.acquire(timeout=20)

                            yield EmbedWithContext(batch)
                            batch = []

                    if tag != 'O' and tag[2:] == ent_type:

                        entity_positions.append(pos)
                        entity_title = link_title

                if len(entity_positions) > 0:

                    sen_part, rel_positions = compute_window(sen, entity_positions)

                    batch.append((entity_title, sen_part, rel_positions))

                    if len(batch) >= batch_size:

                        if embed_semaphore is not None:
                            embed_semaphore.acquire(timeout=20)

                        yield EmbedWithContext(batch)
                        batch = []

    @staticmethod
    def run(embeddings, data_sequence, ent_type, w_size, batch_size, processes, sem=None, start_iteration=0):

        for result in \
                prun(EmbedWithContext._get_all(data_sequence, start_iteration, ent_type, w_size, batch_size, sem),
                     processes=processes, initializer=EmbedWithContext.initialize, initargs=(embeddings,)):

            for _, link_result in result.iterrows():
                yield link_result
