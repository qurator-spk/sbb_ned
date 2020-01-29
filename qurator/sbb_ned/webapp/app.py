import os
import logging
from flask import Flask, send_from_directory, redirect, jsonify, request
from pprint import pprint
# import html
import json
import pandas as pd
# import torch
# from somajo import Tokenizer, SentenceSplitter
#
# from qurator.sbb_ner.models.bert import get_device, model_predict
# from qurator.sbb_ner.ground_truth.data_processor import NerProcessor, convert_examples_to_features
# from qurator.sbb_ner.models.tokenization import BertTokenizer
# from pytorch_pretrained_bert.modeling import (CONFIG_NAME,
#                                               BertConfig,
#                                               BertForTokenClassification)
app = Flask(__name__)

app.config.from_json('config.json' if not os.environ.get('CONFIG') else os.environ.get('CONFIG'))

logger = logging.getLogger(__name__)


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

    parsed = parse_entities(ner)

    pprint(parsed)

    # import ipdb;ipdb.set_trace()

    return "OK", 200


@app.route('/<path:path>')
def send_js(path):
    return send_from_directory('static', path)
