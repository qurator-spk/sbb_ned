import os
import logging
from flask import Flask, send_from_directory, redirect, jsonify, request
from flask_caching import Cache
from hashlib import sha256
# from pprint import pprint
# import html
import json
import pandas as pd
import pickle
import re
import torch
import sqlite3

from qurator.sbb_ned.models.bert import get_device
from qurator.sbb_ned.models.classifier_decider_queue import ClassifierDeciderQueue
from qurator.sbb_ner.models.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import (CONFIG_NAME, BertConfig, BertForSequenceClassification)

from ..embeddings.base import load_embeddings
from ..models.ned_lookup import NEDLookup

from nltk.stem.snowball import SnowballStemmer

app = Flask(__name__)

app.config.from_json('de-config.json' if not os.environ.get('CONFIG') else os.environ.get('CONFIG'))

cache = Cache(app)

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.DEBUG)


class ThreadStore:

    def __init__(self):
        self.ned_lookup = None

        self.model = None

        self.decider = None

        self.classify_decider_queue = None

        self.device = None

        self.tokenizer = None

        self._stemmer = None

        self._entities = None

        self._redirects = None

        self.normalization_map = None

    def get_tokenizer(self):

        if self.tokenizer is not None:

            return self.tokenizer

        model_config = json.load(open(os.path.join(app.config['MODEL_DIR'], "model_config.json"), "r"))

        self.tokenizer = \
            BertTokenizer.from_pretrained(app.config['MODEL_DIR'], do_lower_case=model_config['do_lower'])

        return self.tokenizer

    def get_stemmer(self):

        if self._stemmer is not None:

            return self._stemmer

        self._stemmer = SnowballStemmer(app.config['STEMMER'])

        return self._stemmer

    def get_entities(self):

        if self._entities is not None:
            return self._entities

        with sqlite3.connect(app.config['NED_SQL_FILE']) as con:
            self._entities = pd.read_sql('SELECT * from entities', con).set_index('page_title')

        return self._entities

    def get_redirects(self):

        if self._redirects is not None:
            return self._redirects

        with sqlite3.connect(app.config['NED_SQL_FILE']) as con:
            self._redirects = pd.read_sql('SELECT * from redirects', con).set_index('rd_from_title')

        return self._redirects

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

        if len(app.config['DECIDER_MODEL']) < 1:
            return None

        with open(app.config['DECIDER_MODEL'], 'rb') as fr:
            self.decider = pickle.load(fr)

        return self.decider

    def get_classify_decider_queue(self):

        if self.classify_decider_queue is not None:
            return self.classify_decider_queue

        model, device = self.get_model()
        decider = thread_store.get_decider()

        self.classify_decider_queue = \
            ClassifierDeciderQueue(model, device, decider, app.config['DECISION_THRESHOLD'], self.get_entities(),
                                   app.config['DECIDER_PROCESSES'], app.config['BATCH_SIZE'])

        return self.classify_decider_queue

    def get_lookup_queue(self):

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

    def get_normalization_map(self):

        if self.normalization_map is not None:
            return self.normalization_map

        with sqlite3.connect(app.config['NED_SQL_FILE']) as con:

            table = pd.read_sql('SELECT * from normalization', con)

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


def key_prefix():
    return "{}:{}".format(request.path, sha256(str(request.json).encode('utf-8')).hexdigest())


@app.route('/parse', methods=['GET', 'POST'])
@cache.cached(key_prefix=key_prefix)
def parse_entities():
    ner = request.json

    parsed = dict()

    normalization_map = thread_store.get_normalization_map()

    stemmer = thread_store.get_stemmer()

    redirects = thread_store.get_redirects()

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
                normalized = "".join([normalization_map[c] if c in normalization_map else c for c in entity])
                stem = " ".join([stemmer.stem(p) for p in re.split(' |-|_', normalized)])

                surfaces = {normalized, stem}
                if normalized in redirects.index:
                    surfaces.add(redirects.loc[normalized].rd_title.replace('_', ' '))

                surfaces = list(surfaces)

                logger.debug(str(surfaces))

                try:
                    parsed[entity_id] = {'sentences': [parsed_sent], 'type': ent_type, 'surfaces': surfaces}
                except KeyError as e:
                    logger.error(str(e))

    return jsonify(parsed)


@app.route('/ned', methods=['GET', 'POST'])
@cache.cached(key_prefix=key_prefix)
def ned():
    return_full = bool(request.args.get('return_full', type=int))

    threshold = request.args.get('threshold', default=app.config['DECISION_THRESHOLD'], type=float)

    parsed = request.json

    ned_lookup_queue = thread_store.get_lookup_queue()

    classify_decider_queue = thread_store.get_classify_decider_queue()

    def classify_and_decide_on_lookup_results(job_sequence):

        return classify_decider_queue.run(job_sequence, len(parsed), return_full, threshold)

    ned_result = ned_lookup_queue.run_on_features(parsed, classify_and_decide_on_lookup_results)

    torch.cuda.empty_cache()

    return jsonify(ned_result)


@app.route('/<path:path>')
def send_js(path):
    return send_from_directory('static', path)
