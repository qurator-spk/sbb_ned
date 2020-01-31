import os
import logging
from flask import Flask, send_from_directory, redirect, jsonify, request
from pprint import pprint
# import html
import json
import numpy as np
import pandas as pd

# import torch

import torch
from torch.utils.data import (DataLoader, SequentialSampler, Dataset, TensorDataset)

from collections import OrderedDict

# from somajo import Tokenizer, SentenceSplitter
#
from qurator.sbb_ned.models.bert import get_device, model_predict_compare

# from qurator.sbb_ner.ground_truth.data_processor import NerProcessor, convert_examples_to_features
from qurator.sbb_ner.models.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import (CONFIG_NAME, BertConfig, BertForSequenceClassification)

from ..embeddings.base import load_embeddings
from ..models.ned_lookup import NEDLookup, NEDDataset

app = Flask(__name__)

app.config.from_json('config.json' if not os.environ.get('CONFIG') else os.environ.get('CONFIG'))

logger = logging.getLogger(__name__)


class NEDLookupStore:

    ned_lookup = None

    model = None

    device = None

    tokenizer = None

    def __init__(self):
        pass

    def get_config(self):
        pass

    def get_tokenizer(self):

        if NEDLookupStore.tokenizer is not None:

            return NEDLookupStore.tokenizer

        model_config = json.load(open(os.path.join(app.config['MODEL_DIR'], "model_config.json"), "r"))

        NEDLookupStore.tokenizer = \
            BertTokenizer.from_pretrained(app.config['MODEL_DIR'], do_lower_case=model_config['do_lower'])

        return NEDLookupStore.tokenizer

    @staticmethod
    def get_model():

        if NEDLookupStore.model is not None:
            return NEDLookupStore.model, NEDLookupStore.device

        no_cuda = False if not os.environ.get('USE_CUDA') else os.environ.get('USE_CUDA').lower() == 'false'

        NEDLookupStore.device, n_gpu = get_device(no_cuda=no_cuda)

        config_file = os.path.join(app.config['MODEL_DIR'], CONFIG_NAME)
        #
        model_file = os.path.join(app.config['MODEL_DIR'], app.config['MODEL_FILE'])

        config = BertConfig(config_file)

        NEDLookupStore.model = BertForSequenceClassification(config, num_labels=2)
        # noinspection PyUnresolvedReferences
        NEDLookupStore.model.load_state_dict(torch.load(model_file,
                                                        map_location=lambda storage,
                                                        loc: storage if no_cuda else None))
        # noinspection PyUnresolvedReferences
        NEDLookupStore.model.to(NEDLookupStore.device)
        # noinspection PyUnresolvedReferences
        NEDLookupStore.model.eval()

        return NEDLookupStore.model, NEDLookupStore.device

    def get_lookup(self):

        if NEDLookupStore.ned_lookup is not None:

            return NEDLookupStore.ned_lookup

        embs, dims = load_embeddings(app.config['EMBEDDING_TYPE'])

        embeddings = {'PER': embs, 'LOC': embs, 'ORG': embs}

        NEDLookupStore.ned_lookup =\
            NEDLookup(max_seq_length=app.config['MAX_SEQ_LENGTH'],
                      tokenizer=self.get_tokenizer(),
                      ned_sql_file=app.config['NED_SQL_FILE'],
                      embeddings=embeddings,
                      n_trees=app.config['N_TREES'],
                      distance_measure=app.config['DISTANCE_MEASURE'],
                      entity_index_path=app.config['ENTITY_INDEX_PATH'],
                      search_k=app.config['SEARCH_K'],
                      max_dist=app.config['MAX_DIST'],
                      lookup_processes=app.config['LOOKUP_PROCESSES'],
                      pairing_processes=app.config['PAIRING_PROCESSES'],
                      feature_processes=app.config['FEATURE_PROCESSES'])

        return NEDLookupStore.ned_lookup


ned_lookup_store = NEDLookupStore()


@app.route('/')
def entry():
    return redirect("/index.html", code=302)


def parse_entities(ner):

    parsed = dict()

    for sent in ner:

        entities = []
        entity_types = dict()

        entity = []
        ent_type = None

        for p in sent:

            if len(entity) > 0 and (p['prediction'] == 'O' or p['prediction'].startswith('B-')
                                    or p['prediction'][2:] != ent_type):
                entities += len(entity) * [" ".join(entity)]
                entity_types[" ".join(entity)] = ent_type
                entity = []
                ent_type = None

            if p['prediction'] != 'O':
                entity.append(p['word'])

                if ent_type is None:
                    ent_type = p['prediction'][2:]
            else:
                entities.append("")

        if len(entity) > 0:
            entities += len(entity) * " ".join(entity)

        parsed_sent = \
            {
                'text': json.dumps([p['word'] for p in sent]),
                'tags': json.dumps([p['prediction'] for p in sent]),
                'entities': json.dumps(entities),
            }

        for ent in list(set(entities)):

            if len(ent) == 0:
                continue

            if ent in parsed:
                parsed[ent]['sentences'].append(parsed_sent)
            else:
                parsed[ent] = {'sentences': [parsed_sent], 'type': entity_types[ent]}

    for k in parsed.keys():

        parsed[k]['sentences'] = pd.DataFrame(parsed[k]['sentences'])
        parsed[k]['sentences']['target'] = k

    return parsed


@app.route('/ned', methods=['GET', 'POST'])
def ned():

    ner = request.json

    model, device = ned_lookup_store.get_model()

    parsed = parse_entities(ner)

    ned_candidates, ned_features = ned_lookup_store.get_lookup().get_features(parsed)

    model_ranking = OrderedDict()

    for candidates, features in zip(ned_candidates, ned_features):

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)

        data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_labels)

        sampler = SequentialSampler(data)

        data_loader = DataLoader(data, sampler=sampler, batch_size=app.config['BATCH_SIZE'])

        decision = model_predict_compare(data_loader, device, model, disable_output=True)

        decision['target'] = [f.guid[1] for f in features]
        decision['surface'] = [f.guid[0] for f in features]

        decision['scores'] = np.log(decision[1] / decision[0])

        ranking = decision.groupby('target').sum().sort_values('scores', ascending=False)

        # import ipdb;ipdb.set_trace()

        model_ranking[decision.surface.unique()[0]] = \
            ranking[['scores']].T.to_dict(orient='records', into=OrderedDict)[0]

    return jsonify(model_ranking)


@app.route('/<path:path>')
def send_js(path):
    return send_from_directory('static', path)
