import os
import logging
from flask import Flask, send_from_directory, redirect, jsonify, request
import html
import json
import torch
from somajo import Tokenizer, SentenceSplitter

from qurator.sbb_ner.models.bert import get_device, model_predict
from qurator.sbb_ner.ground_truth.data_processor import NerProcessor, convert_examples_to_features
from qurator.sbb_ner.models.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import (CONFIG_NAME,
                                              BertConfig,
                                              BertForTokenClassification)
app = Flask(__name__)

app.config.from_json('config.json' if not os.environ.get('CONFIG') else os.environ.get('CONFIG'))

logger = logging.getLogger(__name__)


@app.route('/')
def entry():
    return redirect("/index.html", code=302)


@app.route('/models')
def get_models():
    return jsonify(app.config['MODELS'])


@app.route('/ned', methods=['GET', 'POST'])
def ner_bert_tokens():

    # raw_text = request.json['text']

    output = []

    return jsonify(output)


@app.route('/<path:path>')
def send_js(path):
    return send_from_directory('static', path)
