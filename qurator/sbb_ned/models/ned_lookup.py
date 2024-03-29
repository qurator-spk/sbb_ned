from __future__ import absolute_import, division, print_function

import logging
# import os
import pandas as pd
import inspect

from ..index import LookUpByEmbeddings
from ..embeddings.base import EmbedTask
from .sentence_lookup import SentenceLookup
from .jobs import JobQueue, InfiniteLoop

from qurator.utils.parallel import run as prun
from qurator.utils.parallel import run_unordered as prun_unordered

import json
import traceback

from ..ground_truth.data_processor import ConvertSamples2Features, InputExample

logger = logging.getLogger(__name__)


class EmbedTaskWrapper(EmbedTask):

    def __init__(self, job_id, entity_id, ent_type, sentences, *args, **kwargs):

        self._job_id = job_id
        self._entity_id = entity_id
        self._ent_type = ent_type
        self._sentences = sentences

        super(EmbedTaskWrapper, self).__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):

        if self._entity_id is None:
            return self._job_id, None, None, None, (None, None, None)

        try:
            return self._job_id, self._entity_id, self._ent_type, self._sentences, \
                   super(EmbedTaskWrapper, self).__call__(*args, **kwargs)
        except Exception as e:
            logger.error("Exception in EmbedTaskWrapper: {}: {}".format(e, traceback.format_exc()))

            return self._job_id, None, None, None, (None, None, None)


class LookUpByEmbeddingWrapper(LookUpByEmbeddings):

    def __init__(self, job_id, entity_id, sentences, *args, **kwargs):

        self._job_id = job_id
        self._entity_id = entity_id
        self._sentences = sentences

        super(LookUpByEmbeddingWrapper, self).__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):

        if self._entity_id is None:
            return self._job_id, self._sentences, (None, None)

        try:
            return self._job_id, self._sentences, super(LookUpByEmbeddingWrapper, self).__call__(*args, **kwargs)
        except Exception as e:

            logger.error("Exception in LookUpByEmbeddingWrapper: {}: {}".format(e, traceback.format_exc()))

            return self._job_id, self._sentences, (None, None)


class SentenceLookupWrapper(SentenceLookup):

    def __init__(self, job_id, entity_id, sentences=None, candidates=None, **kwargs):

        if candidates is None:
            candidates = []

        if sentences is None:
            sentences = []

        self._job_id = job_id
        self._entity_id = entity_id
        self._candidates = candidates

        super(SentenceLookupWrapper, self).__init__(sentences, candidates, **kwargs)

    def __call__(self, *args, **kwargs):

        if self._entity_id is None:
            return self._job_id, None, None, None

        try:
            return self._job_id, self._entity_id, self._candidates, \
                   super(SentenceLookupWrapper, self).__call__(*args, **kwargs)
        except Exception as e:
            logger.error("Exception in SentenceLookupWrapper: {}: {}".format(e, traceback.format_exc()))

            return self._job_id, None, None, None


class ConvertSamples2FeaturesWrapper(ConvertSamples2Features):

    def __init__(self, job_id, entity_id, candidate=None, *args, **kwargs):

        self._job_id = job_id
        self._entity_id = entity_id
        self._candidate = candidate

        super(ConvertSamples2FeaturesWrapper, self).__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        if self._entity_id is None:
            return self._job_id, None, None, None

        try:
            return self._job_id, self._entity_id, self._candidate, \
                   super(ConvertSamples2FeaturesWrapper, self).__call__(*args, **kwargs)
        except Exception as e:
            logger.error("Exception in ConvertSamples2FeaturesWrapper: {}: {}".format(e, traceback.format_exc()))

            return self._job_id, None, None, None


class NEDLookup:
    """
    """

    quit = False

    def __init__(self, max_seq_length, tokenizer,
                 ned_sql_file, entities_file, embeddings, n_trees, distance_measure,
                 entity_index_path, entity_types, embed_processes=0,
                 lookup_processes=0, pairing_processes=0, feature_processes=0,
                 max_pairs=1000, split_parts=True, verbose=False, limit=3):

        # logger.info('NEDLookup __init__')

        self._max_seq_length = max_seq_length
        self._tokenizer = tokenizer
        self._ned_sql_file = ned_sql_file

        self._entities_file = entities_file
        self._embeddings = embeddings
        self._n_trees = n_trees
        self._distance_measure = distance_measure
        self._entity_index_path = entity_index_path
        self._entity_types = entity_types

        self._embed_processes = embed_processes
        self._lookup_processes = lookup_processes
        self._pairing_processes = pairing_processes
        self._feature_processes = feature_processes

        self._max_pairs = max_pairs
        self._split_parts = split_parts

        self._queue_entities = JobQueue(result_sequence=self.infinite_feature_sequence(),
                                        name="NEDLookup_entities", min_level=2, verbose=True, limit=limit)

        self._queue_embed = JobQueue(name="NEDLookup_embed", min_level=2, verbose=False, feeder_queue=self._queue_entities)

        self._queue_lookup = JobQueue(name="NEDLookup_lookup", min_level=2, verbose=False, feeder_queue=self._queue_embed)

        self._queue_sentences = JobQueue(name="NEDLookup_sentences", min_level=2, verbose=False, feeder_queue=self._queue_lookup)

        self._queue_pairs = JobQueue(name="NEDLookup_pairs", min_level=2, verbose=False, feeder_queue=self._queue_sentences)

        self._queue_features = JobQueue(name="NEDLookup_features", min_level=2, verbose=False, feeder_queue=self._queue_pairs)

        self._queue_final_output = JobQueue(name="NEDLookup_final_output", min_level=2, verbose=False,
                                            feeder_queue=self._queue_features)

        self._verbose = verbose

    def remove_job(self, job_id):

        self._queue_entities.remove_job(job_id)

        self._queue_embed.remove_job(job_id)

        self._queue_lookup.remove_job(job_id)

        self._queue_sentences.remove_job(job_id)

        self._queue_pairs.remove_job(job_id)

        self._queue_features.remove_job(job_id)

        self._queue_final_output.remove_job(job_id)

    def feeder_queue(self):
        return self._queue_final_output

    def get_entity(self):

        while True:

            job_id, task_info, iter_quit = self._queue_entities.get_next_task()

            if iter_quit:
                return

            if job_id is None:
                continue

            if task_info is None:
                continue

            entity_id, entity_info, params = task_info

            if entity_id is None:
                raise RuntimeError("entity-id is None!!!")

            if self._verbose:
                logger.info("get_entity:{}:{} / {}".format(job_id, entity_id, entity_info['surfaces']))

            yield job_id, entity_id, pd.DataFrame(entity_info['sentences']), entity_info['surfaces'], entity_info['type']

    def get_embed(self):

        for job_id, entity_id, sentences, surfaces, ent_type in self.get_entity():

            self._queue_embed.add_to_job(job_id, (entity_id, sentences, surfaces, ent_type))

            loop = InfiniteLoop(100, "{}:{}".format(inspect.getframeinfo(inspect.currentframe()).filename,
                                                    inspect.getframeinfo(inspect.currentframe()).lineno))

            while loop():
                job_id, task_info, iter_quit = self._queue_embed.get_next_task()

                if iter_quit:
                    return

                if task_info is None:
                    break

                entity_id, sentences, surfaces, ent_type, params = task_info

                if self._verbose:
                    logger.info("get_embed: {}:{}".format(job_id, entity_id))

                yield EmbedTaskWrapper(job_id, entity_id, ent_type, sentences, page_title=entity_id, entity_label=surfaces,
                                       split_parts=self._split_parts, **params)

    def get_lookup(self):

        for job_id, entity_id, ent_type, sentences, (_, embedded, embedding_config) in \
                prun_unordered(self.get_embed(), initializer=EmbedTask.initialize, initargs=(self._embeddings,),
                               processes=self._embed_processes):

            self._queue_lookup.add_to_job(job_id, (entity_id, ent_type, sentences))

            loop = InfiniteLoop(100, "{}:{}".format(inspect.getframeinfo(inspect.currentframe()).filename,
                                                    inspect.getframeinfo(inspect.currentframe()).lineno))

            while loop():
                job_id, task_info, iter_quit = self._queue_lookup.get_next_task()

                if iter_quit:
                    return

                if task_info is None:
                    break

                entity_id, ent_type, sentences, params = task_info
                if self._verbose:
                    logger.info("get_lookup: {}:{}({})".format(job_id, entity_id, params))

                # return all the candidates - filtering is done below
                yield LookUpByEmbeddingWrapper(job_id, entity_id, sentences, page_title=entity_id,
                                               entity_embeddings=embedded, embedding_config=embedding_config,
                                               entity_title=entity_id, entity_type=ent_type,
                                               split_parts=self._split_parts, **params)

    def get_sentence_lookup(self):

        for job_id, sentences, (entity_id, candidates) in \
                prun_unordered(self.get_lookup(), initializer=LookUpByEmbeddings.initialize,
                               initargs=(self._entities_file, self._entity_types, self._n_trees, self._distance_measure,
                                         self._entity_index_path, self._ned_sql_file),
                               processes=self._lookup_processes):

            if len(candidates) == 0:
                self._queue_sentences.add_to_job(job_id, (sentences, entity_id, None))
            else:
                for idx in range(0, len(candidates)):
                    self._queue_sentences.add_to_job(job_id, (sentences, entity_id, candidates.iloc[[idx]]))

            self._queue_sentences.add_to_job(job_id, (sentences, None, None))

            loop = InfiniteLoop(100, "{}:{}".format(inspect.getframeinfo(inspect.currentframe()).filename,
                                                    inspect.getframeinfo(inspect.currentframe()).lineno))

            while loop():
                job_id, task_info, iter_quit = self._queue_sentences.get_next_task()

                if iter_quit:
                    return

                if task_info is None:
                    break

                sentences, entity_id, candidates, params = task_info

                if self._verbose:
                    logger.info("get_sentence_lookup: {}:{}".format(job_id, entity_id))

                yield SentenceLookupWrapper(job_id, entity_id, sentences=sentences, candidates=candidates,
                                            max_pairs=self._max_pairs, **params)

    def get_sentence_pairs(self):

        for job_id, entity_id, candidate, pairs in \
                prun(self.get_sentence_lookup(), initializer=SentenceLookup.initialize,
                     initargs=(self._ned_sql_file, ), processes=self._pairing_processes):

            if entity_id is None:
                # signal entity_id == None
                self._queue_pairs.add_to_job(job_id, (entity_id, None, None))
            else:

                if pairs is None:
                    self._queue_pairs.add_to_job(job_id, (entity_id, candidate, None))
                else:

                    for idx, row in pairs.iterrows():
                        pair = (row.id_a, row.id_b, json.loads(row.sen_a), json.loads(row.sen_b),
                                row.pos_a, row.pos_b, row.end_a, row.end_b, row.label)

                        self._queue_pairs.add_to_job(job_id, (entity_id, candidate, pair))

                        candidate = None

            loop = InfiniteLoop(100, "{}:{}".format(inspect.getframeinfo(inspect.currentframe()).filename,
                                                    inspect.getframeinfo(inspect.currentframe()).lineno))

            while loop():

                job_id, task_info, iter_quit = self._queue_pairs.get_next_task()

                if iter_quit:
                    return

                if task_info is None:
                    break

                entity_id, candidate, pair, params = task_info

                if self._verbose:
                    logger.info("get_sentence_pairs: {}:{}".format(job_id, entity_id))

                yield job_id, entity_id, candidate, pair

    def get_feature_tasks(self):

        for job_id, entity_id, candidate, pair in self.get_sentence_pairs():

            self._queue_features.add_to_job(job_id, (entity_id, candidate, pair))

            loop = InfiniteLoop(100, "{}:{}".format(inspect.getframeinfo(inspect.currentframe()).filename,
                                                    inspect.getframeinfo(inspect.currentframe()).lineno))

            while loop():
                job_id, task_info, iter_quit = self._queue_features.get_next_task()

                if iter_quit:
                    return

                if task_info is None:
                    break

                entity_id, candidate, pair, params = task_info

                if self._verbose:
                    logger.info("get_feature_tasks: {}:{}".format(job_id, entity_id))

                if entity_id is None:
                    # signal entity_id == None
                    yield ConvertSamples2FeaturesWrapper(job_id, entity_id=None, **params)
                    continue

                if pair is None:
                    yield ConvertSamples2FeaturesWrapper(job_id, entity_id, sample=None, **params)
                    continue

                id_a, id_b, sen_a, sen_b, pos_a, pos_b, end_a, end_b, label = pair

                sample = InputExample(guid=(id_a, id_b), text_a=sen_a, text_b=sen_b, pos_a=pos_a, pos_b=pos_b,
                                      end_a=end_a, end_b=end_b, label=label)

                yield ConvertSamples2FeaturesWrapper(job_id, entity_id, candidate=candidate, sample=sample)

    def infinite_feature_sequence(self):

        results = dict()

        for job_id, entity_id, candidate, fe in \
                prun(self.get_feature_tasks(), initializer=ConvertSamples2Features.initialize,
                     initargs=(self._tokenizer, self._max_seq_length), processes=self._feature_processes):

            self._queue_final_output.add_to_job(job_id, (entity_id, candidate, fe))

            loop = InfiniteLoop(100, "{}:{}".format(inspect.getframeinfo(inspect.currentframe()).filename,
                                                    inspect.getframeinfo(inspect.currentframe()).lineno))

            while loop():
                job_id, task_info, iter_quit = self._queue_final_output.get_next_task()

                if iter_quit:
                    return

                if task_info is None:
                    break

                entity_id, candidate, fe, params = task_info

                if self._verbose:
                    logger.info("infinite_feature_sequence: {}:{}".format(job_id, entity_id))

                if job_id not in results:
                    results[job_id] = {'features': [], 'candidates': [], 'entity_id': entity_id}

                if entity_id is None:

                    result = results.pop(job_id)

                    yield job_id, (result['entity_id'], result['features'],
                                   (pd.concat(result['candidates']) if len(result['candidates']) > 0 else []))

                    continue

                if fe is not None:
                    results[job_id]['features'].append(fe)

                if candidate is not None:
                    results[job_id]['candidates'].append(candidate)

    def run_on_features(self, ner_info, func, priority, params=None):

        if params is None:
            params = dict()

        job_main = self._queue_entities.add_job([i for i in ner_info.items()], priority=priority, params=params)

        job_embed = self._queue_embed.add_job([], priority=priority, params=params)

        job_lookup = self._queue_lookup.add_job([], priority=priority, params=params)

        job_sentences = self._queue_sentences.add_job([], priority=priority, params=params)

        job_pairs = self._queue_pairs.add_job([], priority=priority, params=params)

        job_features = self._queue_features.add_job([], priority=priority, params=params)

        job_final_output = self._queue_final_output.add_job([], priority=priority, params=params)

        ret = func(job_main.sequence())

        job_main.remove()

        job_embed.remove()

        job_lookup.remove()

        job_sentences.remove()

        job_pairs.remove()

        job_features.remove()

        job_final_output.remove()

        return ret
