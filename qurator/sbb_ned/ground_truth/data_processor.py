from __future__ import absolute_import, division, print_function

import pandas as pd

import torch

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset, Dataset)
from torch.utils.data.distributed import DistributedSampler

from sklearn.utils import shuffle

from ..index import LookUpBySurface

from qurator.utils.parallel import run as prun

import sqlite3

import itertools

import json


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b, pos_a, pos_b, label):

        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.pos_a = pos_a
        self.pos_b = pos_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, guid, input_ids, input_mask, segment_ids, label_id, tokens):
        self.guid = guid
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.tokens = tokens


class SentenceLookup:

    conn = None

    def __init__(self, good, bad):
        self._good = good
        self._bad = bad

    def __call__(self, *args, **kwargs):

        if len(self._good) == 0:
            return None

        good_sentences = []
        for idx, row in self._good.iterrows():

            sen = pd.read_sql(
                "select sentences.page_title, links.target, sentences.text, sentences.entities from links "
                "join sentences on links.sentence=sentences.id "
                "where links.target=?;", SentenceLookup.conn, params=(row.guessed_title,))

            if len(sen) == 0:
                continue

            good_sentences.append(sen)

        if len(good_sentences) == 0:
            return None

        good_sentences = shuffle(pd.concat(good_sentences))

        if len(good_sentences) < 2:
            return None

        bad_sentences = []
        for idx, row in self._bad.iterrows():

            sen = pd.read_sql(
                "select sentences.page_title, links.target, sentences.text, sentences.entities from links "
                "join sentences on links.sentence=sentences.id "
                "where links.target=?;", SentenceLookup.conn, params=(row.guessed_title,))

            if len(sen) == 0:
                continue

            bad_sentences.append(sen)

        if len(bad_sentences) == 0:
            return None

        bad_sentences = shuffle(pd.concat(bad_sentences))

        if len(bad_sentences) < 2:
            return None

        num = min(20, min(len(good_sentences), len(bad_sentences)))

        good_sentences = good_sentences.iloc[:num]
        bad_sentences = bad_sentences.iloc[:num]

        good_sentences['pos'] = \
            [min([i for i, ent in enumerate(json.loads(good_sentences.iloc[p].entities))
                  if ent == good_sentences.iloc[p].target]) for p in range(len(good_sentences))]

        bad_sentences['pos'] = \
            [min([i for i, ent in enumerate(json.loads(bad_sentences.iloc[p].entities))
                  if ent == bad_sentences.iloc[p].target]) for p in range(len(bad_sentences))]

        combis = [(a, b) for a, b in itertools.combinations(range(num), 2)]

        good_pairs = [(good_sentences.iloc[a].text, good_sentences.iloc[b].text,
                       good_sentences.iloc[a].pos, good_sentences.iloc[b].pos, 1) for a, b in combis]

        bad_pairs = [(good_sentences.iloc[a].text, bad_sentences.iloc[b].text,
                      good_sentences.iloc[a].pos, bad_sentences.iloc[b].pos, -1) for a, b in combis]

        ret = pd.DataFrame(good_pairs + bad_pairs, columns=['sen_a', 'sen_b', 'pos_a', 'pos_b', 'label'])

        return ret

    @staticmethod
    def initialize(ned_sql_file):

        SentenceLookup.conn = sqlite3.connect(ned_sql_file)


class WikipediaDataset(Dataset):
    """
    """

    def __init__(self, epoch_size, label_map, max_seq_length, tokenizer,
                 ned_sql_file, entities_file, embeddings, n_trees, distance_measure, path, search_k, max_dist):

        self._epoch_size = epoch_size
        self._label_map = label_map
        self._max_seq_length = max_seq_length
        self._tokenizer = tokenizer
        self._ned_sql_file = ned_sql_file
        self._entities = pd.read_pickle(entities_file)
        self._embeddings = embeddings
        self._n_trees = n_trees
        self._distance_measure = distance_measure
        self._path = path
        self._search_k = search_k
        self._max_dist = max_dist

        self._sequence = self.get_features()
        self._counter = 0

        pass

    def random_entity(self):

        while True:

            self._entities = shuffle(self._entities)

            for idx, row in self._entities.iterrows():

                yield idx, row.TYPE

    def get_random_lookup(self):

        for page_title, ent_type in self.random_entity():

            yield LookUpBySurface(page_title, entity_surface_parts=[page_title], entity_title=page_title,
                                  entity_type=ent_type, split_parts=True)

    def get_sentence_lookup(self):

        for entity_title, ranking in \
                prun(self.get_random_lookup(), initializer=LookUpBySurface.initialize,
                     initargs=(self._embeddings, self._n_trees, self._distance_measure, self._path, self._search_k,
                               self._max_dist), processes=2):

            good = ranking.loc[ranking.guessed_title == entity_title].copy()
            bad = ranking.loc[ranking.guessed_title != entity_title].copy()

            yield SentenceLookup(good, bad)

    def get_sentence_pairs(self):

        for pairs in prun(self.get_sentence_lookup(), initializer=SentenceLookup.initialize,
                          initargs=(self._ned_sql_file,), processes=5):

            if pairs is None:
                continue

            for idx, row in pairs.iterrows():

                yield json.loads(row.sen_a), json.loads(row.sen_b), row.pos_a, row.pos_b, row.label

    def get_features(self):

        for sen_a, sen_b, pos_a, pos_b, label in self.get_sentence_pairs():

            sample = InputExample(guid="%s-%s" % (self._ned_sql_file, self._counter),
                                  text_a=sen_a, text_b=sen_b, pos_a=pos_a, pos_b=pos_b, label=label)
            self._counter += 1

            for fe in convert_examples_to_features(sample, self._label_map, self._max_seq_length, self._tokenizer):

                yield fe

    def __getitem__(self, index):

        del index

        fe = next(self._sequence)

        return torch.tensor(fe.input_ids, dtype=torch.long), torch.tensor(fe.input_mask, dtype=torch.long),\
               torch.tensor(fe.segment_ids, dtype=torch.long), torch.tensor(fe.label_id, dtype=torch.long)

    def __len__(self):

        return int(self._epoch_size)


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, batch_size, local_rank):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, batch_size, local_rank):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    def get_evaluation_file(self):
        raise NotImplementedError()


# class WikipediaNerProcessor(DataProcessor):
#
#     def __init__(self, train_sets, dev_sets, test_sets, gt_file, max_seq_length, tokenizer,
#                  data_epochs, epoch_size, **kwargs):
#         del kwargs
#
#         self._max_seq_length = max_seq_length
#         self._tokenizer = tokenizer
#         self._train_set_file = train_sets
#         self._dev_set_file = dev_sets
#         self._test_set_file = test_sets
#         self._gt_file = gt_file
#         self._data_epochs = data_epochs
#         self._epoch_size = epoch_size
#
#     def get_train_examples(self, batch_size, local_rank):
#         """See base class."""
#
#         return self._make_data_loader(self._train_set_file, batch_size, local_rank)
#
#     def get_dev_examples(self, batch_size, local_rank):
#         """See base class."""
#
#         return self._make_data_loader(self._dev_set_file, batch_size, local_rank)
#
#     def get_labels(self):
#         """See base class."""
#
#         labels = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "X", "[CLS]", "[SEP]"]
#
#         return {label: i for i, label in enumerate(labels)}
#
#     def get_evaluation_file(self):
#         dev_set_name = os.path.splitext(os.path.basename(self._dev_set_file))[0]
#
#         return "eval_results-{}.pkl".format(dev_set_name)
#
#     def _make_data_loader(self, set_file, batch_size, local_rank):
#         del local_rank
#
#         data = WikipediaDataset(set_file=set_file, gt_file=self._gt_file,
#                                 data_epochs=self._data_epochs, epoch_size=self._epoch_size,
#                                 label_map=self.get_labels(), tokenizer=self._tokenizer,
#                                 max_seq_length=self._max_seq_length)
#
#         sampler = SequentialSampler(data)
#
#         return DataLoader(data, sampler=sampler, batch_size=batch_size)


# class NEDProcessor(DataProcessor):
#
#     def __init__(self, train_sets, dev_sets, test_sets, max_seq_length, tokenizer,
#                  label_map=None, gt=None, gt_file=None, **kwargs):
#
#         del kwargs
#
#         self._max_seg_length = max_seq_length
#         self._tokenizer = tokenizer
#         self._train_sets = set(train_sets.split('|')) if train_sets is not None else set()
#         self._dev_sets = set(dev_sets.split('|')) if dev_sets is not None else set()
#         self._test_sets = set(test_sets.split('|')) if test_sets is not None else set()
#
#         self._gt = gt
#
#         if self._gt is None:
#             self._gt = pd.read_pickle(gt_file)
#
#         self._label_map = label_map
#
#         print('TRAIN SETS: ', train_sets)
#         print('DEV SETS: ', dev_sets)
#         print('TEST SETS: ', test_sets)
#
#     def get_train_examples(self, batch_size, local_rank):
#         """See base class."""
#
#         return self.make_data_loader(
#                             self.create_examples(self._read_lines(self._train_sets), "train"), batch_size, local_rank,
#                             self.get_labels(), self._max_seg_length, self._tokenizer)
#
#     def get_dev_examples(self, batch_size, local_rank):
#         """See base class."""
#         return self.make_data_loader(
#                         self.create_examples(self._read_lines(self._dev_sets), "dev"), batch_size, local_rank,
#                         self.get_labels(), self._max_seg_length, self._tokenizer)
#
#     def get_labels(self):
#         """See base class."""
#
#         if self._label_map is not None:
#             return self._label_map
#
#         gt = self._gt
#         gt = gt.loc[gt.dataset.isin(self._train_sets.union(self._dev_sets).union(self._test_sets))]
#
#         labels = sorted(gt.tag.unique().tolist()) + ["X", "[CLS]", "[SEP]"]
#
#         self._label_map = {label: i for i, label in enumerate(labels, 1)}
#
#         self._label_map['UNK'] = 0
#
#         return self._label_map
#
#     def get_evaluation_file(self):
#
#         return "eval_results-{}.pkl".format("-".join(sorted(self._dev_sets)))
#
#     @staticmethod
#     def create_examples(lines, set_type):
#
#         for i, (sentence, label) in enumerate(lines):
#             guid = "%s-%s" % (set_type, i)
#             text_a = sentence
#             text_b = None
#             label = label
#
#             yield InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
#
#     @staticmethod
#     def make_data_loader(examples, batch_size, local_rank, label_map, max_seq_length, tokenizer, features=None,
#                          sequential=False):
#
#         if features is None:
#             features = [fe for ex in examples for fe in
#                         convert_examples_to_features(ex, label_map, max_seq_length, tokenizer)]
#
#         all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
#         all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
#         all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
#         all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
#
#         data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
#
#         if local_rank == -1:
#             if sequential:
#                 train_sampler = SequentialSampler(data)
#             else:
#                 train_sampler = RandomSampler(data)
#         else:
#             if sequential:
#                 train_sampler = SequentialSampler(data)
#             else:
#                 train_sampler = DistributedSampler(data)
#
#         return DataLoader(data, sampler=train_sampler, batch_size=batch_size)
#
#     def _read_lines(self, sets):
#
#         gt = self._gt
#         gt = gt.loc[gt.dataset.isin(sets)]
#
#         data = list()
#         for i, sent in gt.groupby('nsentence'):
#
#             sent = sent.sort_values('nword', ascending=True)
#
#             data.append((sent.word.tolist(), sent.tag.tolist()))
#
#         return data


def convert_examples_to_features(example, label_map, max_seq_len, tokenizer):
    """
    :param example: instance of InputExample
    :param label_map: Maps labels like B-ORG ... to numbers (ids).
    :param max_seq_len: Maximum length of sequences to be delivered to the model.
    :param tokenizer: BERT-Tokenizer
    :return:
    """
    tokens = []
    labels = []

    for i, word in enumerate(example.text_a):  # example.text_a is a sequence of words

        token = tokenizer.tokenize(word)
        tokens.extend(token)

        label_1 = example.label[i] if i < len(example.label) else 'O'

        for m in range(len(token)):  # a word might have been split into several tokens
            if m == 0:
                labels.append(label_1)
            else:
                labels.append("X")

    start_pos = 0
    while start_pos < len(tokens):

        window_len = min(max_seq_len - 2, len(tokens) - start_pos)  # -2 since we also need [CLS] and [SEP]

        # Make sure that we do not split the sentence within a word.
        while window_len > 1 and start_pos + window_len < len(tokens) and\
                tokens[start_pos + window_len].startswith('##'):
            window_len -= 1

        if window_len == 1:
            window_len = min(max_seq_len - 2, len(tokens) - start_pos)

        token_window = tokens[start_pos:start_pos+window_len]
        start_pos += window_len

        augmented_tokens = ["[CLS]"] + token_window + ["[SEP]"]

        input_ids = tokenizer.convert_tokens_to_ids(augmented_tokens) + max(0, max_seq_len - len(augmented_tokens))*[0]

        input_mask = [1] * len(augmented_tokens) + max(0, max_seq_len - len(augmented_tokens))*[0]

        segment_ids = [0] + len(token_window) * [0] + [0] + max(0, max_seq_len - len(augmented_tokens))*[0]

        label_ids = [label_map["[CLS]"]] + [label_map[labels[i]] for i in range(len(token_window))] + \
                    [label_map["[SEP]"]] + max(0, max_seq_len - len(augmented_tokens)) * [0]

        assert len(input_ids) == max_seq_len
        assert len(input_mask) == max_seq_len
        assert len(segment_ids) == max_seq_len
        assert len(label_ids) == max_seq_len

        yield InputFeatures(guid=example.guid, input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids,
                            label_id=label_ids, tokens=augmented_tokens)

