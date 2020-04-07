from __future__ import absolute_import, division, print_function

import logging
# import os
import pandas as pd

from ..index import LookUpBySurface

from qurator.utils.parallel import run as prun
# from qurator.utils.parallel import run_unordered as prun_unordered
from multiprocessing import Semaphore

from math import sqrt, ceil

import sqlite3

import itertools

import json

import threading

from ..ground_truth.data_processor import ConvertSamples2Features, InputExample

# logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
#                     datefmt='%m/%d/%Y %H:%M:%S',
#                     level=logging.INFO)

logger = logging.getLogger(__name__)


class SentenceLookup:
    ned_sql_file = None
    connection_map = None

    def __init__(self, found_sentences, candidates, max_pairs=0):

        self._found_sentences = found_sentences

        self._max_pairs = max_pairs  # number of sentence pair comparisons per candidate

        self._use_found = int(min(ceil(sqrt(self._max_pairs)), len(self._found_sentences)))

        self._candidates = candidates

    def select_sentences(self):

        limit = int(float(self._max_pairs) / float(self._use_found))

        tmp = []
        for _, candidate in self._candidates.iterrows():

            s = pd.read_sql(
                "select sentences.id,sentences.page_title, links.target, sentences.text, sentences.entities "
                "from links join sentences on links.sentence=sentences.id where links.target=? limit ?;",
                SentenceLookup.get_db(), params=(candidate.guessed_title, limit))

            if len(s) == 0:
                logger.debug('1')
                continue

            tmp.append(s)

        if len(tmp) == 0:
            return pd.DataFrame([], columns=['page_title', 'target', 'text', 'entities'])

        tmp = pd.concat(tmp)

        tmp = tmp.loc[~tmp.page_title.str.startswith('Liste ')]

        return tmp

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

        if len(self._found_sentences) == 0:
            return None

        candidate_sentences = self.select_sentences()

        candidate_sentences = candidate_sentences.loc[SentenceLookup.is_valid_sentence(candidate_sentences)]

        if len(candidate_sentences) == 0:
            logger.debug('No candidate sentences. Number of candidates: {}'.format(len(self._candidates)))
            return None

        self._found_sentences['pos'], self._found_sentences['end'] = \
            SentenceLookup.locate_entities(self._found_sentences)

        candidate_sentences['pos'], candidate_sentences['end'] = \
            SentenceLookup.locate_entities(candidate_sentences)

        combis = [(a, b) for a, b in
                  itertools.product(range(self._use_found), range(len(candidate_sentences)))]

        check_pairs = [(self._found_sentences.iloc[a].target, candidate_sentences.iloc[b].target,
                        self._found_sentences.iloc[a].text, candidate_sentences.iloc[b].text,
                        self._found_sentences.iloc[a].pos, candidate_sentences.iloc[b].pos,
                        self._found_sentences.iloc[a].end, candidate_sentences.iloc[b].end, 0) for a, b in combis]

        check_pairs = pd.DataFrame(check_pairs,
                                   columns=['id_a', 'id_b', 'sen_a', 'sen_b', 'pos_a', 'pos_b', 'end_a', 'end_b',
                                            'label'])

        return check_pairs

    @staticmethod
    def initialize(ned_sql_file):

        SentenceLookup.ned_sql_file = ned_sql_file
        SentenceLookup.connection_map = dict()

    @staticmethod
    def get_db():

        thid = threading.current_thread().ident

        conn = SentenceLookup.connection_map.get(thid)

        if conn is None:
            logger.info('Create database connection: {}'.format(SentenceLookup.ned_sql_file))

            conn = sqlite3.connect(SentenceLookup.ned_sql_file)

            SentenceLookup.connection_map[thid] = conn

        return conn


class LookUpBySurfaceWrapper(LookUpBySurface):

    def __init__(self, sentences, *args, **kwargs):

        self._sentences = sentences

        super(LookUpBySurfaceWrapper, self).__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):

        return self._sentences, super(LookUpBySurfaceWrapper, self).__call__(*args, **kwargs)


class SentenceLookupWrapper(SentenceLookup):

    def __init__(self, entity_id, sequence_number=None, sentences=None, candidates=None, **kwargs):

        if candidates is None:
            candidates = []

        if sentences is None:
            sentences = []

        self._entity_id = entity_id
        self._sequence_number = sequence_number
        self._candidates = candidates

        super(SentenceLookupWrapper, self).__init__(sentences, candidates, **kwargs)

    def __call__(self, *args, **kwargs):

        if self._sequence_number is None:
            return self._entity_id, None, None, None

        return self._entity_id, self._sequence_number, self._candidates,\
               super(SentenceLookupWrapper, self).__call__(*args, **kwargs)


class ConvertSamples2FeaturesWrapper(ConvertSamples2Features):

    def __init__(self, entity_id, pair_number=None, candidate=None, *args, **kwargs):

        self._entity_id = entity_id
        self._pair_number = pair_number
        self._candidate = candidate

        super(ConvertSamples2FeaturesWrapper, self).__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):

        if self._pair_number is None:

            return self._entity_id, self._candidate, self._pair_number, None

        return self._entity_id, self._candidate, self._pair_number, \
               super(ConvertSamples2FeaturesWrapper, self).__call__(*args, **kwargs)


class NEDLookup:
    """
    """

    quit = False

    def __init__(self, max_seq_length, tokenizer,
                 ned_sql_file, entities_file, embeddings, n_trees, distance_measure,
                 entity_index_path, search_k, max_dist,
                 lookup_processes=0, pairing_processes=0, feature_processes=0, max_candidates=20,
                 max_pairs=1000, split_parts=True):

        logger.info('NEDLookup __init__')

        self._max_seq_length = max_seq_length
        self._tokenizer = tokenizer
        self._ned_sql_file = ned_sql_file

        self._entities_file = entities_file
        self._embeddings = embeddings
        self._n_trees = n_trees
        self._distance_measure = distance_measure
        self._entity_index_path = entity_index_path
        self._search_k = search_k
        self._max_dist = max_dist

        self._lookup_processes = lookup_processes
        self._pairing_processes = pairing_processes
        self._feature_processes = feature_processes

        self._process_queue = []
        self._process_queue_sem = Semaphore(0)

        self._main_sem = Semaphore(1)

        self._sequence = self.infinite_feature_sequence()

        self._max_candidates = max_candidates
        self._max_pairs = max_pairs
        self._split_parts = split_parts

    def get_entity(self):

        while True:

            if not self.wait(self._process_queue_sem):
                return

            entities = self._process_queue.pop()

            for entity_id, v in entities.items():

                logger.debug(entity_id)

                sentences = pd.DataFrame(v['sentences'])

                yield entity_id, sentences, v['surface'], v['type']

    def get_lookup(self):

        for entity_id, sentences, surface, ent_type in self.get_entity():

            yield LookUpBySurfaceWrapper(sentences, page_title=entity_id, entity_surface_parts=[surface],
                                         entity_title=entity_id, entity_type=ent_type, split_parts=self._split_parts,
                                         max_candidates=self._max_candidates)

    def get_sentence_lookup(self):

        for sentences, (entity_id, candidates) in \
                prun(self.get_lookup(), initializer=LookUpBySurface.initialize,
                     initargs=(self._entities_file, self._embeddings, self._n_trees, self._distance_measure,
                               self._entity_index_path, self._search_k, self._max_dist),
                     processes=self._lookup_processes):

            for idx in range(0, len(candidates)):
                yield SentenceLookupWrapper(entity_id, idx, sentences, candidates.iloc[[idx]],
                                            max_pairs=self._max_pairs)

            yield SentenceLookupWrapper(entity_id)

    def get_sentence_pairs(self):

        pair_number = 0
        for entity_id, sequence_number, candidate, pairs in \
                prun(self.get_sentence_lookup(), initializer=SentenceLookup.initialize,
                     initargs=(self._ned_sql_file, ), processes=self._pairing_processes):

            if sequence_number is None:

                pair_number = 0
                yield entity_id, None, None, None

                continue

            if pairs is None:
                continue

            for idx, row in pairs.iterrows():

                pair = (row.id_a, row.id_b, json.loads(row.sen_a), json.loads(row.sen_b),
                        row.pos_a, row.pos_b, row.end_a, row.end_b, row.label)

                yield entity_id, candidate, pair_number, pair

                candidate = None

                pair_number += 1

    def get_feature_tasks(self):

        for entity_id, candidate, pair_number, pair in self.get_sentence_pairs():

            if pair_number is None:
                yield ConvertSamples2FeaturesWrapper(entity_id)
                continue

            id_a, id_b, sen_a, sen_b, pos_a, pos_b, end_a, end_b, label = pair

            sample = InputExample(guid=(id_a, id_b), text_a=sen_a, text_b=sen_b, pos_a=pos_a, pos_b=pos_b,
                                  end_a=end_a, end_b=end_b, label=label)

            yield ConvertSamples2FeaturesWrapper(entity_id, pair_number=pair_number, candidate=candidate, sample=sample)

    def infinite_feature_sequence(self):

        features = []
        candidates = []

        for entity_id, candidate, pair_number, fe in \
                prun(self.get_feature_tasks(), initializer=ConvertSamples2Features.initialize,
                     initargs=(self._tokenizer, self._max_seq_length), processes=self._feature_processes):

            if pair_number is None:
                yield entity_id, features, pd.concat(candidates) if len(candidates) > 0 else []

                features = []
                candidates = []
                continue

            if fe is not None:
                features.append(fe)

            if candidate is not None:
                candidates.append(candidate)

    @staticmethod
    def wait(sem=None):

        while True:
            if sem is not None and sem.acquire(timeout=10):
                return True

            if NEDLookup.quit:
                return False

            if sem is None:
                return True

    def run_on_features(self, ner_info, func):

        self._main_sem.acquire()

        self._process_queue.append(ner_info)

        self._process_queue_sem.release()

        def job_sequence():

            for _ in range(len(ner_info)):

                entity_id, fe, cand = next(self._sequence)

                yield entity_id, fe, cand

        ret = func(job_sequence())

        self._main_sem.release()

        return ret
