import os
import threading
import logging
from flask import Flask, send_from_directory, redirect, jsonify, request
# from pprint import pprint
# import html
import json
import numpy as np
import pandas as pd
import sqlite3
import pickle
# import torch

import torch
from torch.utils.data import (DataLoader, SequentialSampler, TensorDataset)

from collections import OrderedDict

from qurator.sbb_ned.models.bert import get_device, model_predict_compare
from qurator.sbb_ned.models.decider import predict
from qurator.sbb_ned.models.decider import features as make_decider_features
from qurator.utils.parallel import run as prun

from qurator.sbb_ner.models.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import (CONFIG_NAME, BertConfig, BertForSequenceClassification)

from ..embeddings.base import load_embeddings
from ..models.ned_lookup import NEDLookup

from tqdm import tqdm as tqdm
from multiprocessing import Semaphore


app = Flask(__name__)

app.config.from_json('config.json' if not os.environ.get('CONFIG') else os.environ.get('CONFIG'))

logger = logging.getLogger(__name__)


class ThreadStore:

    def __init__(self):
        self.ned_lookup = None

        self.model = None

        self.decider = None

        self.classify_decider_queue = None

        self.device = None

        self.tokenizer = None

        self.connection_map = None

        self.normalization_map = None

    def get_tokenizer(self):

        if self.tokenizer is not None:

            return self.tokenizer

        model_config = json.load(open(os.path.join(app.config['MODEL_DIR'], "model_config.json"), "r"))

        self.tokenizer = \
            BertTokenizer.from_pretrained(app.config['MODEL_DIR'], do_lower_case=model_config['do_lower'])

        return self.tokenizer

    def get_model(self):

        if self.model is not None:
            return self.model, self.device

        no_cuda = False if not os.environ.get('USE_CUDA') else os.environ.get('USE_CUDA').lower() == 'false'

        self.device, n_gpu = get_device(no_cuda=no_cuda)

        config_file = os.path.join(app.config['MODEL_DIR'], CONFIG_NAME)
        #
        model_file = os.path.join(app.config['MODEL_DIR'], app.config['MODEL_FILE'])

        config = BertConfig(config_file)

        self.model = BertForSequenceClassification(config, num_labels=2)
        # noinspection PyUnresolvedReferences
        self.model.load_state_dict(torch.load(model_file,
                                              map_location=lambda storage, loc: storage if no_cuda else None))
        # noinspection PyUnresolvedReferences
        self.model.to(self.device)
        # noinspection PyUnresolvedReferences
        self.model.eval()

        return self.model, self.device

    def get_decider(self):

        if self.decider is not None:
            return self.decider

        with open(app.config['DECIDER_MODEL'], 'rb') as fr:
            self.decider = pickle.load(fr)

        return self.decider

    def get_classify_decider_queue(self):

        if self.classify_decider_queue is not None:
            return self.classify_decider_queue

        self.classify_decider_queue = ClassifierDeciderQueue()

        return self.classify_decider_queue

    def get_lookup(self):

        if self.ned_lookup is not None:

            return self.ned_lookup

        no_cuda = False if not os.environ.get('USE_CUDA') else os.environ.get('USE_CUDA').lower() == 'false'

        embs = load_embeddings(app.config['EMBEDDING_TYPE'], model_path=app.config["EMBEDDING_MODEL_PATH"],
                               pooling_operation=app.config['POOLING'], no_cuda=no_cuda)

        embeddings = {'PER': embs, 'LOC': embs, 'ORG': embs}

        self.ned_lookup =\
            NEDLookup(max_seq_length=app.config['MAX_SEQ_LENGTH'],
                      tokenizer=self.get_tokenizer(),
                      ned_sql_file=app.config['NED_SQL_FILE'],
                      entities_file=app.config['ENTITIES_FILE'],
                      embeddings=embeddings,
                      n_trees=app.config['N_TREES'],
                      distance_measure=app.config['DISTANCE_MEASURE'],
                      entity_index_path=app.config['ENTITY_INDEX_PATH'],
                      search_k=app.config['SEARCH_K'],
                      max_dist=app.config['MAX_DIST'],
                      lookup_processes=app.config['LOOKUP_PROCESSES'],
                      pairing_processes=app.config['PAIRING_PROCESSES'],
                      feature_processes=app.config['FEATURE_PROCESSES'],
                      max_candidates=app.config['MAX_CANDIDATES'],
                      max_pairs=app.config['MAX_PAIRS'],
                      split_parts=app.config['SPLIT_PARTS'])

        return self.ned_lookup

    def get_wiki_db(self):

        if self.connection_map is None:
            self.connection_map = dict()

        thid = threading.current_thread().ident

        conn = self.connection_map.get(thid)

        if conn is None:
            logger.info('Create database connection: {}'.format(app.config['WIKIPEDIA_SQL_FILE']))

            conn = sqlite3.connect(app.config['WIKIPEDIA_SQL_FILE'])

            self.connection_map[thid] = conn

        return conn

    def get_normalization_map(self):

        if self.normalization_map is not None:
            return self.normalization_map

        table = pd.read_pickle(app.config['NORMALIZATION_TABLE'])

        self.normalization_map = {row['unicode']: row['base'] for _, row in table.iterrows()}

        return self.normalization_map


thread_store = ThreadStore()


@app.route('/')
def entry():
    return redirect("/index.html", code=302)


def parse_sentence(sent, normalization_map):
    entities = []
    entity_types = []
    entity = []
    ent_type = None

    for p in sent:

        if len(entity) > 0 and (p['prediction'] == 'O' or p['prediction'].startswith('B-')
                                or p['prediction'][2:] != ent_type):
            entities += len(entity) * [" ".join(entity)]
            entity_types += len(entity) * [ent_type]
            entity = []
            ent_type = None

        if p['prediction'] != 'O':
            entity.append(p['word'])

            if ent_type is None:
                ent_type = p['prediction'][2:]
        else:
            entities.append("")
            entity_types.append("")

    if len(entity) > 0:
        entities += len(entity) * [" ".join(entity)]
        entity_types += len(entity) * [ent_type]

    entity_ids = ["{}-{}".format(entity, ent_type) for entity, ent_type in zip(entities, entity_types)]

    text_json = json.dumps(
        ["".join([normalization_map[c] if c in normalization_map else c for c in p['word']]) for p in sent])

    tags_json = json.dumps([p['prediction'] for p in sent])

    entities_json = json.dumps(entity_ids)

    return entity_ids, entities, entity_types, text_json, tags_json, entities_json


@app.route('/parse', methods=['GET', 'POST'])
def parse_entities():
    ner = request.json

    parsed = dict()

    normalization_map = thread_store.get_normalization_map()

    for sent in ner:

        entity_ids, entities, entity_types, text_json, tags_json, entities_json = \
            parse_sentence(sent, normalization_map)

        already_processed = set()

        for entity_id, entity, ent_type in zip(entity_ids, entities, entity_types):

            if len(entity) == 0:
                continue

            if entity_id in already_processed:
                continue

            already_processed.add(entity_id)

            parsed_sent = {'text': text_json, 'tags': tags_json, 'entities': entities_json, 'target': entity_id}

            if entity_id in parsed:
                parsed[entity_id]['sentences'].append(parsed_sent)
            else:
                surface = "".join([normalization_map[c] if c in normalization_map else c for c in entity])

                try:
                    parsed[entity_id] = {'sentences': [parsed_sent], 'type': ent_type, 'surface': surface}
                except KeyError as e:
                    logger.error(str(e))

    return jsonify(parsed)


class DeciderTask:

    decider = None

    def __init__(self, entity_id, decision, candidates, quantiles, rank_intervalls, threshold, return_full=False):

        self._entity_id = entity_id
        self._decision = decision
        self._candidates = candidates
        self._quantiles = quantiles
        self._rank_intervalls = rank_intervalls
        self._threshold = threshold
        self._return_full = return_full

    def __call__(self, *args, **kwargs):

        if self._candidates is None:
            return self._entity_id, None

        wiki_db_conn = thread_store.get_wiki_db()

        decider_features = make_decider_features(self._decision, self._candidates, self._quantiles,
                                                 self._rank_intervalls)

        prediction = predict(decider_features, DeciderTask.decider)

        ranking = prediction[(prediction.proba_1 > self._threshold) |
                             (prediction.guessed_title == self._decision.target.unique()[0])]. \
            sort_values(['proba_1', 'case_rank_max'], ascending=[False, True]). \
            set_index('guessed_title')

        result = dict()

        if len(ranking) > 0:
            ranking['wikidata'] = [DeciderTask.get_wk_id(wiki_db_conn, k) for k, _ in ranking.iterrows()]

            result['ranking'] = [i for i in ranking[['proba_1', 'wikidata']].T.to_dict(into=OrderedDict).items()]

        if self._return_full:
            self._candidates['wikidata'] = [DeciderTask.get_wk_id(wiki_db_conn, v.guessed_title)
                                            for _, v in self._candidates.iterrows()]

            decision = self._decision.merge(self._candidates[['guessed_title', 'wikidata']],
                                            left_on='guessed_title', right_on='guessed_title')

            result['decision'] = json.loads(decision.to_json(orient='split'))
            result['candidates'] = json.loads(self._candidates.to_json(orient='split'))

        return self._entity_id, result

    @staticmethod
    def get_wk_id(db_conn, page_title):

        _wk_id = pd.read_sql("select page_props.pp_value from page_props "
                             "join page on page.page_id==page_props.pp_page "
                             "where page.page_title==? and page.page_namespace==0 "
                             "and page_props.pp_propname=='wikibase_item';", db_conn,
                             params=(page_title,))

        if _wk_id is None or len(_wk_id) == 0:
            return None

        return _wk_id.iloc[0][0]

    @staticmethod
    def initialize(decider):

        DeciderTask.decider = decider


class ClassifierDeciderQueue:

    quit = False

    def __init__(self):

        logger.info('ClassifierDeciderQueue __init__')

        self._process_queue = []
        self._process_queue_sem = Semaphore(0)
        self._main_sem = Semaphore(1)

        self._model, self._device = thread_store.get_model()

        self._decider = thread_store.get_decider()
        self._rank_intervalls = np.linspace(0.001, 0.1, 100)
        self._quantiles = np.linspace(0.1, 1, 10)

        self._threshold = app.config['DECISION_THRESHOLD']
        self._return_full = False

        self._sequence = self.process_sequence()

    def run(self, job_sequence, len_sequence, return_full, threshold):

        self._main_sem.acquire()

        _threshold, self._threshold = self._threshold, threshold
        _return_full, self._return_full = self._return_full, return_full

        self._process_queue.append((job_sequence, len_sequence))

        self._process_queue_sem.release()

        ret = next(self._sequence)

        self._threshold = _threshold
        self._return_full = _return_full

        self._main_sem.release()

        return ret

    def process_sequence(self):

        complete_result = OrderedDict()

        for eid, result in prun(self.get_decider_tasks(), initializer=DeciderTask.initialize,
                                initargs=(self._decider,), processes=app.config['DECIDER_PROCESSES']):

            if eid is None:
                yield complete_result
                complete_result = OrderedDict()
                continue

            if result is None:
                continue

            complete_result[eid] = result

    def get_decider_tasks(self):

        while True:

            if not self.wait(self._process_queue_sem):
                return

            job_sequence, len_sequence = self._process_queue.pop()

            for entity_id, features, candidates in tqdm(job_sequence, total=len_sequence):

                if len(candidates) == 0:
                    yield DeciderTask(entity_id, None, None, quantiles=None, rank_intervalls=None, threshold=None)
                    continue

                all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
                all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
                all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
                all_labels = torch.tensor([f.label for f in features], dtype=torch.long)

                data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_labels)

                sampler = SequentialSampler(data)

                data_loader = DataLoader(data, sampler=sampler, batch_size=app.config['BATCH_SIZE'])

                decision = model_predict_compare(data_loader, self._device, self._model, disable_output=True)
                decision['guessed_title'] = [f.guid[1] for f in features]
                decision['target'] = [f.guid[0] for f in features]
                decision['scores'] = np.log(decision[1] / decision[0])

                assert len(decision.target.unique()) == 1

                yield DeciderTask(entity_id, decision, candidates, self._quantiles, self._rank_intervalls,
                                  self._threshold, self._return_full)

            yield DeciderTask(entity_id=None, decision=None, candidates=None, quantiles=None, rank_intervalls=None,
                              threshold=None)

    @staticmethod
    def wait(sem=None):

        while True:
            if sem is not None and sem.acquire(timeout=10):
                return True

            if ClassifierDeciderQueue.quit:
                return False

            if sem is None:
                return True


@app.route('/ned', methods=['GET', 'POST'])
def ned():
    return_full = request.args.get('return_full', default=False, type=bool)

    threshold = request.args.get('threshold', default=app.config['DECISION_THRESHOLD'], type=float)

    parsed = request.json

    ned_lookup = thread_store.get_lookup()

    classify_decider_queue = thread_store.get_classify_decider_queue()

    ned_result = ned_lookup.run_on_features(parsed,
                                            lambda job_sequence:
                                            classify_decider_queue.run(job_sequence, len(parsed),
                                                                       return_full, threshold))
    return jsonify(ned_result)


@app.route('/<path:path>')
def send_js(path):
    return send_from_directory('static', path)
