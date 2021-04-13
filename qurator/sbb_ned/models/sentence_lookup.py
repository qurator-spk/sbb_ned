from __future__ import absolute_import, division, print_function
import pandas as pd

from math import sqrt, ceil

import sqlite3

import itertools

import json

import threading


class SentenceLookup:
    ned_sql_file = None
    connection_map = None

    def __init__(self, found_sentences, candidates, max_pairs=0, **kwargs):

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
                # logger.debug('1')
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
            # logger.debug('No candidate sentences. Number of candidates: {}'.format(len(self._candidates)))
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
            # logger.info('Create database connection: {}'.format(SentenceLookup.ned_sql_file))

            conn = sqlite3.connect(SentenceLookup.ned_sql_file)

            SentenceLookup.connection_map[thid] = conn

        return conn