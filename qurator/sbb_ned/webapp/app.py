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
from torch.utils.data import (DataLoader, SequentialSampler, TensorDataset)

from collections import OrderedDict

from qurator.sbb_ned.models.bert import get_device, model_predict_compare

from qurator.sbb_ner.models.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import (CONFIG_NAME, BertConfig, BertForSequenceClassification)

from ..embeddings.base import load_embeddings
from ..models.ned_lookup import NEDLookup

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
                      feature_processes=app.config['FEATURE_PROCESSES'],
                      max_candidates=app.config['MAX_CANDIDATES'],
                      max_pairs=app.config['MAX_PAIRS'])

        return NEDLookupStore.ned_lookup


ned_lookup_store = NEDLookupStore()


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


@app.route('/ned', methods=['GET', 'POST'])
def ned():

    parsed = request.json

    model, device = ned_lookup_store.get_model()

    # parsed = parse_entities(ner)

    model_ranking = OrderedDict()

    def classify_with_bert(entity_id, features, candidates):

        # del candidates

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

        assert len(decision.target.unique()) == 1

        decision['scores'] = np.log(decision[1] / decision[0])

        # decision = decision.query('scores > {}'.format(app.config['DECISION_THRESHOLD']))

        # ranking = decision.groupby('guessed_title').max().sort_values('scores', ascending=False)

        # model_ranking[decision.target.unique()[0]] = \
        #     [i for i in ranking[['scores']].T.to_dict(orient='records', into=OrderedDict)[0].items()]

        #

        threshold = app.config['DECISION_THRESHOLD']

        # noinspection PyUnresolvedReferences
        ranking = [(k, ((g.scores > threshold).sum() - (g.scores < -threshold).sum())) # / len(g))
                   for k, g in decision.groupby('guessed_title')]

        ranking = pd.DataFrame(ranking, columns=['guessed_title', 'sentence_score'])

        ranking = candidates.merge(ranking, on='guessed_title')

        #    sort_values(['len_pa', 'len_guessed', 'sentence_score'], ascending=[False, True, True]).\
        #    reset_index(drop=True)

        # ranking['rank'] = ranking.index

        ranking = ranking.sort_values(['sentence_score', 'rank'], ascending=[False, True])

        # ranking = pd.DataFrame([(k,
        #                          len(g.query('scores > {}'.format(app.config['DECISION_THRESHOLD'])))/len(g)
        #                          )
        #                         for k, g in decision.groupby('target')],
        #                        columns=['target', 'hits']).\

        ranking = ranking.\
            query('sentence_score > 0 or guessed_title=="{}"'.format(decision.target.unique()[0])).\
            set_index('guessed_title')

        # import ipdb;ipdb.set_trace()

        if len(ranking) == 0:
            return

        model_ranking[entity_id] = \
            [i for i in ranking[['sentence_score']].T.to_dict(orient='records', into=OrderedDict)[0].items()]

    ned_lookup_store.get_lookup().run_on_features(parsed, classify_with_bert)

    # import ipdb;ipdb.set_trace()

    return jsonify(model_ranking)


@app.route('/<path:path>')
def send_js(path):
    return send_from_directory('static', path)
