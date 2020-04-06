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


app = Flask(__name__)

app.config.from_json('config.json' if not os.environ.get('CONFIG') else os.environ.get('CONFIG'))

logger = logging.getLogger(__name__)


class ThreadStore:

    ned_lookup = None

    model = None

    decider = None

    device = None

    tokenizer = None

    connection_map = None

    def __init__(self):
        pass

    def get_config(self):
        pass

    @staticmethod
    def get_tokenizer():

        if ThreadStore.tokenizer is not None:

            return ThreadStore.tokenizer

        model_config = json.load(open(os.path.join(app.config['MODEL_DIR'], "model_config.json"), "r"))

        ThreadStore.tokenizer = \
            BertTokenizer.from_pretrained(app.config['MODEL_DIR'], do_lower_case=model_config['do_lower'])

        return ThreadStore.tokenizer

    @staticmethod
    def get_model():

        if ThreadStore.model is not None:
            return ThreadStore.model, ThreadStore.device

        no_cuda = False if not os.environ.get('USE_CUDA') else os.environ.get('USE_CUDA').lower() == 'false'

        ThreadStore.device, n_gpu = get_device(no_cuda=no_cuda)

        config_file = os.path.join(app.config['MODEL_DIR'], CONFIG_NAME)
        #
        model_file = os.path.join(app.config['MODEL_DIR'], app.config['MODEL_FILE'])

        config = BertConfig(config_file)

        ThreadStore.model = BertForSequenceClassification(config, num_labels=2)
        # noinspection PyUnresolvedReferences
        ThreadStore.model.load_state_dict(torch.load(model_file,
                                                     map_location=lambda storage, loc: storage if no_cuda else None))
        # noinspection PyUnresolvedReferences
        ThreadStore.model.to(ThreadStore.device)
        # noinspection PyUnresolvedReferences
        ThreadStore.model.eval()

        return ThreadStore.model, ThreadStore.device

    @staticmethod
    def get_decider():

        if ThreadStore.decider is not None:
            return ThreadStore.decider

        with open(app.config['DECIDER_MODEL'], 'rb') as fr:
            ThreadStore.decider = pickle.load(fr)

        return ThreadStore.decider

    def get_lookup(self):

        no_cuda = False if not os.environ.get('USE_CUDA') else os.environ.get('USE_CUDA').lower() == 'false'

        if ThreadStore.ned_lookup is not None:

            return ThreadStore.ned_lookup

        embs = load_embeddings(app.config['EMBEDDING_TYPE'], model_path=app.config["EMBEDDING_MODEL_PATH"],
                               pooling_operation=app.config['POOLING'], no_cuda=no_cuda)

        embeddings = {'PER': embs, 'LOC': embs, 'ORG': embs}

        ThreadStore.ned_lookup =\
            NEDLookup(max_seq_length=app.config['MAX_SEQ_LENGTH'],
                      tokenizer=ThreadStore.get_tokenizer(),
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

        return ThreadStore.ned_lookup

    @staticmethod
    def get_wiki_db():

        if ThreadStore.connection_map is None:
            ThreadStore.connection_map = dict()

        thid = threading.current_thread().ident

        conn = ThreadStore.connection_map.get(thid)

        if conn is None:
            logger.info('Create database connection: {}'.format(app.config['WIKIPEDIA_SQL_FILE']))

            conn = sqlite3.connect(app.config['WIKIPEDIA_SQL_FILE'])

            ThreadStore.connection_map[thid] = conn

        return conn


thread_store = ThreadStore()


@app.route('/')
def entry():
    return redirect("/index.html", code=302)


@app.route('/parse', methods=['GET', 'POST'])
def parse_entities():
    ner = request.json

    parsed = dict()

    for sent in ner:

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

        parsed_sent = \
            {
                'text': json.dumps([p['word'] for p in sent]),
                'tags': json.dumps([p['prediction'] for p in sent]),
                'entities': json.dumps(entities),
            }

        already_processed = dict()

        for entity, ent_type in zip(entities, entity_types):

            if len(entity) == 0:
                continue

            entity_id = "{}-{}".format(entity, ent_type)

            if entity_id in already_processed:
                continue

            already_processed[entity_id] = 1

            if entity_id in parsed:
                parsed[entity_id]['sentences'].append(parsed_sent)
            else:
                try:
                    parsed[entity_id] = {'sentences': [parsed_sent], 'type': ent_type, 'target': entity}
                except KeyError:
                    import ipdb
                    ipdb.set_trace()

    return jsonify(parsed)


class DeciderTask:

    wiki_db_conn = None
    decider = None
    return_full = None

    def __init__(self, entity_id, decision, candidates, quantiles, rank_intervalls, threshold):

        self._entity_id = entity_id
        self._decision = decision
        self._candidates = candidates
        self._quantiles = quantiles
        self._rank_intervalls = rank_intervalls
        self._threshold = threshold

    def __call__(self, *args, **kwargs):

        if self._candidates is None:
            return self._entity_id, None

        def get_wk_id(db_conn, page_title):

            _wk_id = pd.read_sql("select page_props.pp_value from page_props "
                                 "join page on page.page_id==page_props.pp_page "
                                 "where page.page_title==? and page.page_namespace==0 "
                                 "and page_props.pp_propname=='wikibase_item';", db_conn,
                                 params=(page_title,))

            if _wk_id is None or len(_wk_id) == 0:
                return None

            return _wk_id.iloc[0][0]

        decider_features = make_decider_features(self._decision, self._candidates, self._quantiles,
                                                 self._rank_intervalls)

        prediction = predict(decider_features, DeciderTask.decider)

        ranking = prediction[(prediction.proba_1 > self._threshold) |
                             (prediction.guessed_title == self._decision.target.unique()[0])]. \
            sort_values(['proba_1', 'case_rank_max'], ascending=[False, True]). \
            set_index('guessed_title')

        result = dict()

        if len(ranking) > 0:
            ranking['wikidata'] = [get_wk_id(DeciderTask.wiki_db_conn, k) for k, _ in ranking.iterrows()]

            result['ranking'] = [i for i in ranking[['proba_1', 'wikidata']].T.to_dict(into=OrderedDict).items()]

        if DeciderTask.return_full:
            self._candidates['wikidata'] = [get_wk_id(DeciderTask.wiki_db_conn, v.guessed_title)
                                            for _, v in self._candidates.iterrows()]

            decision = self._decision.merge(self._candidates[['guessed_title', 'wikidata']],
                                            left_on='guessed_title', right_on='guessed_title')

            result['decision'] = json.loads(decision.to_json(orient='split'))
            result['candidates'] = json.loads(self._candidates.to_json(orient='split'))

        return self._entity_id, result

    @staticmethod
    def initialize(decider, return_full):

        DeciderTask.return_full = return_full
        DeciderTask.decider = decider
        DeciderTask.wiki_db_conn = ThreadStore.get_wiki_db()


@app.route('/ned', methods=['GET', 'POST'])
def ned():
    return_full = request.args.get('return_full', default=False, type=bool)

    threshold = request.args.get('threshold', default=app.config['DECISION_THRESHOLD'], type=float)

    rank_intervalls = np.linspace(0.001, 0.1, 100)
    quantiles = np.linspace(0.1, 1, 10)

    parsed = request.json

    model, device = thread_store.get_model()

    decider = thread_store.get_decider()

    def classify_with_bert(job_sequence):

        def make_decider_tasks():
            for entity_id, features, candidates in job_sequence:

                if len(candidates) == 0:
                    yield DeciderTask(entity_id, None, None, quantiles, rank_intervalls, threshold)
                    continue

                all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
                all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
                all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
                all_labels = torch.tensor([f.label for f in features], dtype=torch.long)

                data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_labels)

                sampler = SequentialSampler(data)

                data_loader = DataLoader(data, sampler=sampler, batch_size=app.config['BATCH_SIZE'])

                decision = model_predict_compare(data_loader, device, model, disable_output=True)
                decision['guessed_title'] = [f.guid[1] for f in features]
                decision['target'] = [f.guid[0] for f in features]
                decision['scores'] = np.log(decision[1] / decision[0])

                assert len(decision.target.unique()) == 1

                yield DeciderTask(entity_id, decision, candidates, quantiles, rank_intervalls, threshold)

        complete_result = OrderedDict()

        for eid, result in tqdm(prun(make_decider_tasks(), initializer=DeciderTask.initialize,
                                initargs=(decider, return_full), processes=app.config['DECIDER_PROCESSES']),
                                total=len(parsed)):
            if result is None:
                continue

            complete_result[eid] = result

        return complete_result

    ned_result = thread_store.get_lookup().run_on_features(parsed, classify_with_bert)

    return jsonify(ned_result)


@app.route('/<path:path>')
def send_js(path):
    return send_from_directory('static', path)
