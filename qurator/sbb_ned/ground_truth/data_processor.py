from __future__ import absolute_import, division, print_function

import logging
import os
import pandas as pd
# import numpy as np

import torch

from torch.utils.data import (DataLoader, SequentialSampler, Dataset)

# from torch.utils.data import (RandomSampler, TensorDataset)
# from torch.utils.data.distributed import DistributedSampler

from sklearn.utils import shuffle

from ..index import LookUpBySurface

# from qurator.utils.parallel import run as prun
from qurator.utils.parallel import run_unordered as prun_unordered
from multiprocessing import Semaphore

import sqlite3

import itertools

import json


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b, pos_a, pos_b, end_a, end_b, label):

        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.pos_a = pos_a
        self.pos_b = pos_b
        self.end_a = end_a
        self.end_b = end_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, guid, input_ids, input_mask, segment_ids,  tokens, label):
        self.guid = guid
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.tokens = tokens
        self.label = label


class SentenceLookup:

    conn = None
    sentence_subset = None
    max_samples = 20

    def __init__(self, good, bad):
        self._good = good
        self._bad = bad

    @staticmethod
    def select_sentences(targets, ssubset):

        tmp = []
        for i, r in targets.iterrows():

            s = pd.read_sql(
                "select sentences.id,sentences.page_title, links.target, sentences.text, sentences.entities "
                "from links join sentences on links.sentence=sentences.id where links.target=?;",
                SentenceLookup.conn, params=(r.guessed_title,)).set_index('id').sort_index()

            if ssubset is not None:
                s = pd.merge(s, ssubset, left_index=True, right_index=True)

            if len(s) == 0:
                logger.debug('1')
                continue

            tmp.append(s)

        if len(tmp) == 0:
            return []

        tmp = pd.concat(tmp)

        tmp = tmp.loc[~tmp.page_title.str.startswith('Liste ')]

        return shuffle(tmp)

    @staticmethod
    def locate_entities(sent):

        starts = []
        ends = []

        for _, row in sent.iterrows():
            ent = json.loads(row.entities)
            target = row.target

            parts = [(k, len(list(g))) for k, g in itertools.groupby(ent)]

            pos = end_pos = max_len = 0
            for i, (entity, entity_len) in enumerate(parts):

                if entity != target:
                    continue

                if entity_len <= max_len:
                    continue

                max_len = entity_len
                pos = sum([l for _, l in parts[0:i]])
                end_pos = pos + entity_len

            starts.append(pos)
            ends.append(end_pos)

        return starts, ends

    @staticmethod
    def is_valid_sentence(sent):

        valid = []

        for _, row in sent.iterrows():
            text = json.loads(row.text)

            if text[0].lower() in {'#redirect', '#weiterleitung'}:
                valid.append(False)
                continue

            valid.append(True)

        return valid

    def __call__(self, *args, **kwargs):

        if len(self._good) == 0:
            # print('4')
            return None

        good_sentences = SentenceLookup.select_sentences(self._good, SentenceLookup.sentence_subset)

        if len(good_sentences) < 2:
            logger.debug('2, len(good): {}'.format(len(self._good)))
            return None

        good_sentences = good_sentences.loc[SentenceLookup.is_valid_sentence(good_sentences)]

        if len(good_sentences) < 2:
            logger.debug('2, len(good): {}'.format(len(self._good)))
            return None

        bad_sentences = SentenceLookup.select_sentences(self._bad, SentenceLookup.sentence_subset)

        if len(bad_sentences) < 2:
            logger.debug('3, len(bad): {}'.format(len(self._bad)))
            return None

        bad_sentences = bad_sentences.loc[SentenceLookup.is_valid_sentence(bad_sentences)]

        if len(bad_sentences) < 2:
            logger.debug('3, len(bad): {}'.format(len(self._bad)))
            return None

        num = min(SentenceLookup.max_samples, min(len(good_sentences), len(bad_sentences)))

        good_sentences = good_sentences.iloc[:num]
        bad_sentences = bad_sentences.iloc[:num]

        good_sentences['pos'], good_sentences['end'] = SentenceLookup.locate_entities(good_sentences)
        bad_sentences['pos'], bad_sentences['end'] = SentenceLookup.locate_entities(bad_sentences)

        good_sentences = good_sentences.reset_index()
        bad_sentences = bad_sentences.reset_index()

        combis = [(a, b) for a, b in itertools.combinations(range(len(good_sentences)), 2)]

        good_pairs = [(good_sentences.iloc[a].id, good_sentences.iloc[b].id,
                       good_sentences.iloc[a].text, good_sentences.iloc[b].text,
                       good_sentences.iloc[a].pos, good_sentences.iloc[b].pos,
                       good_sentences.iloc[a].end, good_sentences.iloc[b].end, 1) for a, b in combis]

        combis = [(a, b) for a, b in itertools.product(range(len(good_sentences)), range(len(bad_sentences)))]

        bad_pairs = [(good_sentences.iloc[a].id, bad_sentences.iloc[b].id,
                      good_sentences.iloc[a].text, bad_sentences.iloc[b].text,
                      good_sentences.iloc[a].pos, bad_sentences.iloc[b].pos,
                      good_sentences.iloc[a].end, bad_sentences.iloc[b].end, 0) for a, b in combis]

        if len(bad_pairs) > len(good_pairs):
            bad_pairs = bad_pairs[0:len(good_pairs)]
        else:
            good_pairs = good_pairs[0:len(bad_pairs)]

        ret = pd.DataFrame(good_pairs + bad_pairs,
                           columns=['id_a', 'id_b', 'sen_a', 'sen_b', 'pos_a', 'pos_b', 'end_a', 'end_b', 'label'])
        return shuffle(ret)

    @staticmethod
    def initialize(ned_sql_file, sentence_subset):

        SentenceLookup.conn = sqlite3.connect(ned_sql_file)

        SentenceLookup.sentence_subset = sentence_subset.set_index('id').sort_index()


class WikipediaDataset(Dataset):
    """
    """

    quit = False

    def __init__(self, size, max_seq_length, tokenizer,
                 ned_sql_file, entities, embeddings, n_trees, distance_measure, entity_index_path, search_k, max_dist,
                 sentence_subset=None, bad_count=10, lookup_processes=0, pairing_processes=0):

        self._size = size
        self._max_seq_length = max_seq_length
        self._tokenizer = tokenizer
        self._ned_sql_file = ned_sql_file
        self._entities = entities
        self._embeddings = embeddings
        self._n_trees = n_trees
        self._distance_measure = distance_measure
        self._entity_index_path = entity_index_path
        self._search_k = search_k
        self._max_dist = max_dist
        self._sentence_subset = sentence_subset

        self._bad_count = bad_count
        self._max_bad_count = 50

        self._sequence = self.get_features()
        self._counter = 0
        self._lookup_processes = lookup_processes
        self._pairing_processes = pairing_processes
        self._lookup_sem = Semaphore(100)
        self._convert_sem = Semaphore(1000)

    def random_entity(self):

        while True:

            self._entities = shuffle(self._entities)

            for idx, row in self._entities.iterrows():

                yield idx, row.TYPE

    def get_random_lookup(self):

        for page_title, ent_type in self.random_entity():

            if WikipediaDataset.quit:
                return

            while True:
                if self._lookup_sem.acquire(timeout=10):
                    break

                if WikipediaDataset.quit:
                    return

            yield LookUpBySurface(page_title, entity_surface_parts=[page_title], entity_title=page_title,
                                  entity_type=ent_type, split_parts=True)

    def get_sentence_lookup(self):

        for entity_title, ranking in \
                prun_unordered(self.get_random_lookup(), initializer=LookUpBySurface.initialize,
                               initargs=(self._embeddings, self._n_trees, self._distance_measure,
                                         self._entity_index_path, self._search_k, self._max_dist),
                               processes=self._lookup_processes):

            if WikipediaDataset.quit:
                break

            good = ranking.loc[ranking.guessed_title == entity_title].copy()
            bad = ranking.loc[ranking.guessed_title != entity_title].copy()

            if len(good) == 0:  # There aren't any hits ... skip.
                logger.debug('0')
                self._lookup_sem.release()
                continue

            # we want to have at least bad_count bad examples but also at most max_bad_count examples.
            nbad = max(self._bad_count, min(self._max_bad_count, good['rank'].min()))

            bad = bad.iloc[0:nbad]

            yield SentenceLookup(good, bad)

            del good
            del bad

    def get_sentence_pairs(self):

        for pairs in prun_unordered(self.get_sentence_lookup(), initializer=SentenceLookup.initialize,
                                    initargs=(self._ned_sql_file, self._sentence_subset),
                                    processes=self._pairing_processes):

            if WikipediaDataset.quit:
                break

            if pairs is None:
                self._lookup_sem.release()
                continue

            for idx, row in pairs.iterrows():

                if WikipediaDataset.quit:
                    break

                yield row.id_a, row.id_b,\
                      json.loads(row.sen_a), json.loads(row.sen_b), \
                      row.pos_a, row.pos_b, \
                      row.end_a, row.end_b, \
                      row.label

            self._lookup_sem.release()
            del pairs

    def get_feature_tasks(self):

        for id_a, id_b, sen_a, sen_b, pos_a, pos_b, end_a, end_b, label in self.get_sentence_pairs():

            if WikipediaDataset.quit:
                break

            while True:
                if self._convert_sem.acquire(timeout=10):
                    break

                if WikipediaDataset.quit:
                    return

            sample = InputExample(guid="%s-%s" % (self._ned_sql_file, "{}-{}".format(id_a, id_b)),
                                  text_a=sen_a, text_b=sen_b, pos_a=pos_a, pos_b=pos_b,
                                  end_a=end_a, end_b=end_b, label=label)
            self._counter += 1

            yield ConvertSamples2Features(sample)

            del sample

    def get_features(self):

        for features in prun_unordered(self.get_feature_tasks(), initializer=ConvertSamples2Features.initialize,
                                       initargs=(self._tokenizer, self._max_seq_length), processes=10):

            self._convert_sem.release()

            if features is None:
                continue

            yield features

            del features

    def __getitem__(self, index):

        del index

        fe = next(self._sequence)

        return torch.tensor(fe.input_ids, dtype=torch.long), torch.tensor(fe.input_mask, dtype=torch.long),\
               torch.tensor(fe.segment_ids, dtype=torch.long), torch.tensor(fe.label, dtype=torch.long)

    def __len__(self):

        return int(self._size)


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, batch_size, local_rank):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, batch_size, local_rank):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, batch_size, local_rank):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    @staticmethod
    def num_labels():
        raise NotImplementedError()

    def get_evaluation_file(self):
        raise NotImplementedError()


class WikipediaNEDProcessor:

    class Internal(DataProcessor):

        def __init__(self, train_subset, dev_subset, test_subset, train_size, dev_size, test_size, ned_sql_file,
                     **kwargs):

            self._train_subset = train_subset
            self._dev_subset = dev_subset
            self._test_subset = test_subset

            self._train_size = train_size
            self._dev_size = dev_size
            self._test_size = test_size

            self._ned_sql_file = ned_sql_file

            self._kwargs = kwargs

        def get_train_examples(self, batch_size, local_rank):
            """See base class."""

            return self._make_data_loader(self._train_subset, self._train_size, batch_size, local_rank)

        def get_dev_examples(self, batch_size, local_rank):
            """See base class."""

            return self._make_data_loader(self._dev_subset, self._dev_size, batch_size, local_rank)

        def get_test_examples(self, batch_size, local_rank):
            """See base class."""

            return self._make_data_loader(self._test_subset, self._test_size, batch_size, local_rank)

        @staticmethod
        def num_labels():
            return 2

        def get_evaluation_file(self):
            evaluation_file = os.path.splitext(os.path.basename(self._ned_sql_file))[0]

            return "eval_results-{}.pkl".format(evaluation_file)

        def _make_data_loader(self, subset, size, batch_size, local_rank):
            del local_rank

            data = WikipediaDataset(size=size, sentence_subset=subset, ned_sql_file=self._ned_sql_file, **self._kwargs)

            sampler = SequentialSampler(data)

            return DataLoader(data, sampler=sampler, batch_size=batch_size)

    def __init__(self, *args,  **kwargs):

        self._args = args
        self._kwargs = kwargs

    def __enter__(self):

        return WikipediaNEDProcessor.Internal(*self._args, **self._kwargs)

    def __exit__(self, exc_type, exc_val, exc_tb):

        WikipediaDataset.quit = True


class ConvertSamples2Features:

    tokenizer = None
    max_seq_len = 0

    def __init__(self, sample=None):

        self._sample = sample

    def __call__(self, *args, **kwargs):

        if self._sample is None:
            return None

        return convert_examples_to_features(self._sample, ConvertSamples2Features.max_seq_len,
                                            ConvertSamples2Features.tokenizer)

    @staticmethod
    def initialize(tokenizer, max_seq_len):

        ConvertSamples2Features.tokenizer = tokenizer
        ConvertSamples2Features.max_seq_len = max_seq_len


def convert_examples_to_features(example, max_seq_len, tokenizer, mark_entities=False):
    """
    :param example: instance of InputExample
    :param max_seq_len: Maximum length of sequences to be delivered to the model.
    :param tokenizer: BERT-Tokenizer
    :return:
    """

    def make_tokens(text, entity_pos, entity_end):

        tmp = []
        entity_token_pos = 0
        entity_len = 0

        for i, word in enumerate(text):

            word_tokens = tokenizer.tokenize(word)

            entity_token_pos = len(tmp) if i == entity_pos else entity_token_pos
            entity_len = entity_len + len(word_tokens) if entity_pos <= i < entity_end else entity_len

            tmp.extend(word_tokens)

        return tmp, entity_token_pos, entity_len

    full_tokens_a, pos_a, len_entity_a = make_tokens(example.text_a, example.pos_a, example.end_a)
    full_tokens_b, pos_b, len_entity_b = make_tokens(example.text_b, example.pos_b, example.end_b)

    total_len = len(full_tokens_a) + len(full_tokens_b)

    a_start = pos_a
    a_end = pos_a + len_entity_a

    b_start = pos_b
    b_end = pos_b + len_entity_b

    def max_len_reached():
        return a_end - a_start + b_end - b_start >= max_seq_len - 2  # -2 since we also need [CLS] and [SEP]

    while True:

        a_start = a_start - 1 if a_start > 0 else a_start

        if max_len_reached():
            break

        b_start = b_start - 1 if b_start > 0 else b_start

        if max_len_reached():
            break

        a_end = a_end + 1 if a_end < len(full_tokens_a) else a_end

        if max_len_reached():
            break

        b_end = b_end + 1 if b_end < len(full_tokens_b) else b_end

        if max_len_reached():
            break

        if a_end - a_start + b_end - b_start >= total_len:
            break

    tokens_a = full_tokens_a[a_start:a_end]
    tokens_b = full_tokens_b[b_start:b_end]

    len_pre_context_a = pos_a - a_start
    len_pre_context_b = pos_b - b_start

    len_post_context_a = len(tokens_a) - len_pre_context_a - len_entity_a
    len_post_context_b = len(tokens_b) - len_pre_context_b - len_entity_b

    augmented_tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b

    input_ids = tokenizer.convert_tokens_to_ids(augmented_tokens) + max(0, max_seq_len - len(augmented_tokens))*[0]

    input_mask = [1] * len(augmented_tokens) + max(0, max_seq_len - len(augmented_tokens))*[0]

    entity_a_id = 2 if mark_entities else 0
    entity_b_id = 2 if mark_entities else 1

    segment_ids = [0] + len_pre_context_a*[0] + len_entity_a*[entity_a_id] + len_post_context_a*[0] +\
                  [1] + len_pre_context_b*[1] + len_entity_b*[entity_b_id] + len_post_context_b*[1] +\
                  max(0, max_seq_len - len(augmented_tokens)) * [0]

    try:
        assert len(input_ids) == max_seq_len
        assert len(input_mask) == max_seq_len
        assert len(segment_ids) == max_seq_len
    except AssertionError:
        logger.error('AssertionError, convert_examples_to_features')
        return None

    return InputFeatures(guid=example.guid, input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids,
                         tokens=augmented_tokens, label=example.label)
