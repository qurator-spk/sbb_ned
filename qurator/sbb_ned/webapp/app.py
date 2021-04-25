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
import multiprocessing as mp

from qurator.sbb_ned.models.classifier_decider_queue import ClassifierDeciderQueue
from qurator.sbb_ner.models.tokenization import BertTokenizer

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

        self._embeddings = None

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

        self.get_embeddings()  # DO NOT REMOVE!

        self._sem = mp.Semaphore(1)

    def get_tokenizer(self):

        if self.tokenizer is not None:
            return self.tokenizer

        model_config = json.load(open(os.path.join(app.config['MODEL_DIR'], "model_config.json"), "r"))

        self.tokenizer = \
            BertTokenizer.from_pretrained(app.config['MODEL_DIR'], do_lower_case=model_config['do_lower'])

        return self.tokenizer

    def get_stemmer(self):

        with self._sem:
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

        with self._sem:
            if self._redirects is not None:
                return self._redirects

            with sqlite3.connect(app.config['NED_SQL_FILE']) as con:
                self._redirects = pd.read_sql('SELECT * from redirects', con).set_index('rd_from_title')

            return self._redirects

    def get_decider(self):

        if self.decider is not None:
            return self.decider

        if len(app.config['DECIDER_MODEL']) < 1:
            return None

        with open(app.config['DECIDER_MODEL'], 'rb') as fr:
            self.decider = pickle.load(fr)

        return self.decider

    def get_classify_decider_queue(self):

        lookup = self.get_lookup_queue()

        with self._sem:
            if self.classify_decider_queue is not None:
                return self.classify_decider_queue

            no_cuda = False if not os.environ.get('USE_CUDA') else os.environ.get('USE_CUDA').lower() == 'false'

            self.classify_decider_queue = \
                ClassifierDeciderQueue(no_cuda=no_cuda, model_dir=app.config['MODEL_DIR'],
                                       model_file=app.config['MODEL_FILE'], decider=thread_store.get_decider(),
                                       entities=self.get_entities(), decider_processes=app.config['DECIDER_PROCESSES'],
                                       classifier_processes=app.config['EVALUATOR_PROCESSES'],
                                       batch_size=app.config['BATCH_SIZE'],
                                       feeder_queue=lookup.feeder_queue())

            return self.classify_decider_queue

    def get_embeddings(self):

        if self._embeddings is not None:
            return self._embeddings

        no_cuda = False if not os.environ.get('USE_CUDA') else os.environ.get('USE_CUDA').lower() == 'false'

        self._embeddings = \
            load_embeddings(app.config['EMBEDDING_TYPE'], model_path=app.config["EMBEDDING_MODEL_PATH"],
                            pooling_operation=app.config['POOLING'], layers=app.config["EMBEDDING_LAYERS"],
                            no_cuda=no_cuda)

        return self._embeddings

    def get_lookup_queue(self):

        with self._sem:
            if self.ned_lookup is not None:
                return self.ned_lookup

            self.ned_lookup = \
                NEDLookup(max_seq_length=app.config['MAX_SEQ_LENGTH'],
                          tokenizer=self.get_tokenizer(),
                          ned_sql_file=app.config['NED_SQL_FILE'],
                          entities_file=app.config['ENTITIES_FILE'],
                          embeddings=self.get_embeddings(),
                          n_trees=app.config['N_TREES'],
                          distance_measure=app.config['DISTANCE_MEASURE'],
                          entity_index_path=app.config['ENTITY_INDEX_PATH'],
                          entity_types=['PER', 'LOC', 'ORG'],
                          search_k=app.config['SEARCH_K'],
                          max_dist=app.config['MAX_DIST'],
                          embed_processes=app.config['EMBED_PROCESSES'],
                          lookup_processes=app.config['LOOKUP_PROCESSES'],
                          pairing_processes=app.config['PAIRING_PROCESSES'],
                          feature_processes=app.config['FEATURE_PROCESSES'],
                          max_candidates=app.config['MAX_CANDIDATES'],
                          max_pairs=app.config['MAX_PAIRS'],
                          split_parts=app.config['SPLIT_PARTS'])

            return self.ned_lookup

    def get_normalization_map(self):

        with self._sem:
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
    valid_tags = {'O', 'B-PER', 'B-LOC', 'B-ORG', 'I-PER', 'I-LOC', 'I-ORG'}

    for p in sent:

        p['prediction'] = p['prediction'] if p['prediction'] in valid_tags else "O"

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

                try:
                    parsed[entity_id] = {'sentences': [parsed_sent], 'type': ent_type, 'surfaces': surfaces}
                except KeyError as e:
                    logger.error(str(e))

    return jsonify(parsed)


@app.route('/ned', methods=['GET', 'POST'])
@cache.cached(key_prefix=key_prefix)
def ned():
    return_full = bool(request.args.get('return_full', type=int))
    priority = request.args.get('priority', type=int, default=1)

    threshold = request.args.get('threshold', default=app.config['DECISION_THRESHOLD'], type=float)

    parsed = request.json

    ned_lookup_queue = thread_store.get_lookup_queue()

    classify_decider_queue = thread_store.get_classify_decider_queue()

    def classify_and_decide_on_lookup_results(job_sequence):
        return classify_decider_queue.run(job_sequence, len(parsed), return_full, threshold, priority=priority)

    ned_result = ned_lookup_queue.run_on_features(parsed, classify_and_decide_on_lookup_results, priority=priority)

    torch.cuda.empty_cache()

    return jsonify(ned_result)


@app.route('/<path:path>')
def send_js(path):
    return send_from_directory('static', path)
