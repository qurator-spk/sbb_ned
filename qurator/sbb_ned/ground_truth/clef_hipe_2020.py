import click
import re
from io import StringIO
import numpy as np
import pandas as pd
import unicodedata
from tqdm import tqdm
import json
import logging
import glob
# from CLEF2020.clef_evaluation import get_results
from qurator.utils.parallel import run as prun
from ..models.decider import features

from somajo import Tokenizer, SentenceSplitter
import sqlite3

logger = logging.getLogger(__name__)


def read_clef(clef_file):

    with open(clef_file, 'r') as f:

        sentence_splitter = SentenceSplitter()

        docs = []
        segments = []
        text_part = []
        header = None
        urls = []

        def make_segement():

            nonlocal docs, segments, text_part

            if len(text_part) == 0:
                return

            tmp = None
            # noinspection PyBroadException
            try:
                tmp = pd.read_csv(StringIO(header + "".join(text_part)), sep='\t', comment='#', quoting=3)
            except:
                import ipdb
                ipdb.set_trace()

            tmp = tmp.reset_index().rename(columns={'index': 'TOKEN_ID'})

            tmp['url_id'] = len(docs)
            tmp['segment_id'] = len(segments)

            segments.append(tmp)

            text_part = []

        def make_doc():

            nonlocal docs, segments, sentence_splitter

            doc = pd.concat(segments)

            sentences = sentence_splitter.split(doc.TOKEN.astype(str).to_list())
            doc['TOKEN_ID'] = [i for s in sentences for i in range(len(s))]
            
            docs.append(doc)
            segments = []

        for line in tqdm(f):

            if header is None:
                header = "\t".join(line.split()) + '\n'
                continue

            if not line.startswith('#'):
                text_part.append(line)

            if re.match(r'#\s+segment_iiif_link\s+=.*', line):

                make_segement()

            if re.match(r'#\s+document_id\s+=.*', line):

                make_segement()

                urls.append(line)
                if len(segments) > 0:
                    make_doc()

        make_segement()
        make_doc()

        return urls, pd.concat(docs).reset_index(drop=True)


@click.command()
@click.argument('clef-file', type=click.Path(exists=True), required=True, nargs=1)
@click.argument('tsv-file', type=click.Path(), required=True, nargs=1)
def clef2tsv(clef_file, tsv_file):

    out_columns = ['No.', 'TOKEN', 'NE-TAG', 'NE-EMB', 'ID', 'url_id', 'left', 'right', 'top', 'bottom']

    entity_types = ['PER', 'LOC', 'ORG', 'O']

    urls, df = read_clef(clef_file)

    df['NE-COARSE-LIT'] = df['NE-COARSE-LIT'].str[0:5].str.upper()

    df['NE-NESTED'] = df['NE-NESTED'].str[0:5].str.upper()

    df['left'] = df['right'] = df['top'] = df['bottom'] = 0

    # rename columns such that they match the neat columns.
    df = df.rename(columns={'TOKEN_ID': 'No.', 'TOKEN': 'TOKEN', 'NE-COARSE-LIT': 'NE-TAG', 'NE-NESTED': 'NE-EMB',
                            'NEL-LIT': 'ID', })

    df.loc[~df['NE-TAG'].str[2:5].isin(entity_types), 'NE-TAG'] = 'O'
    df.loc[~df['NE-EMB'].str[2:5].isin(entity_types), 'NE-EMB'] = 'O'

    # make sure that there aren't any control characters in the TOKEN column.
    # Since that would lead to problems later on.
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        df.loc[idx, 'TOKEN'] = "".join([c if unicodedata.category(c) != 'Cc' else '' for c in str(row.TOKEN)])

    # remove rows that have an empty TOKEN.
    df = df.loc[df.TOKEN.str.len() > 0]

    df = df[out_columns]

    pd.DataFrame([], columns=out_columns).to_csv(tsv_file, sep="\t", quoting=3, index=False)

    for (_, part), url in zip(df.groupby('url_id', sort=False, as_index=False), urls):

        with open(tsv_file, 'a') as f:
            f.write(url)

        part.to_csv(tsv_file, sep="\t", quoting=3, index=False, mode="a", header=False)


@click.command()
@click.argument('tsv-file', type=click.Path(exists=True), required=True, nargs=1)
@click.argument('clef-gs-file', type=click.Path(), required=True, nargs=1)
@click.argument('out-clef-file', type=click.Path(), required=True, nargs=1)
def tsv2clef(tsv_file, clef_gs_file, out_clef_file):

    out_columns = ['TOKEN', 'NE-COARSE-LIT', 'NE-COARSE-METO', 'NE-FINE-LIT', 'NE-FINE-METO', 'NE-FINE-COMP',
                   'NE-NESTED', 'NEL-LIT', 'NEL-METO', 'MISC']

    tsv = pd.read_csv(tsv_file, sep='\t', comment='#', quoting=3)
    tsv.loc[tsv.TOKEN.isnull(), 'TOKEN'] = ""

    urls, tsv_gs = read_clef(clef_gs_file)
    tsv_gs['TOKEN'] = tsv_gs.TOKEN.astype(str)

    tsv_out = []

    pd.DataFrame([], columns=out_columns).to_csv(out_clef_file, sep="\t", quoting=3, index=False)

    def write_segment():

        nonlocal tsv_out, out_columns

        tsv_out = pd.DataFrame(tsv_out).\
            rename(columns={'NE-TAG': 'NE-COARSE-LIT', 'NE-EMB': 'NE-NESTED', 'ID': 'NEL-LIT'})

        tsv_out['NE-COARSE-LIT'] = tsv_out['NE-COARSE-LIT'].str.replace('-PER', '-pers')
        tsv_out['NE-COARSE-LIT'] = tsv_out['NE-COARSE-LIT'].str.replace('-LOC', '-loc')
        tsv_out['NE-COARSE-LIT'] = tsv_out['NE-COARSE-LIT'].str.replace('-ORG', '-org')

        tsv_out['NE-NESTED'] = tsv_out['NE-NESTED'].str.replace('-PER', '-pers')
        tsv_out['NE-NESTED'] = tsv_out['NE-NESTED'].str.replace('-LOC', '-loc')
        tsv_out['NE-NESTED'] = tsv_out['NE-NESTED'].str.replace('-ORG', '-org')

        tsv_out['NE-COARSE-METO'] = tsv_out['NE-COARSE-LIT']
        tsv_out['NE-FINE-LIT'] = 'O'
        tsv_out['NE-FINE-METO'] = 'O'
        tsv_out['NE-FINE-COMP'] = 'O'
        tsv_out['NEL-METO'] = '-'
        tsv_out['MISC'] = '-'

        tsv_out = tsv_out[out_columns]

        with open(out_clef_file, 'a') as f:
            f.write("# segment_iiif_link = _\n")

        tsv_out.to_csv(out_clef_file, sep="\t", index=False, mode='a', header=False)

        tsv_out = []

    seq_out = tsv.iterrows()

    _, row_out = next(seq_out, (None, None))

    segment_id = None
    url_id = None

    for _, row_gs in tqdm(tsv_gs.iterrows(), total=len(tsv_gs)):

        cur_token = ""
        ne_tags = set()
        nel_ids = set()

        if row_out is None:
            break

        while row_gs.TOKEN != cur_token and row_out is not None:

            if not row_gs.TOKEN.startswith(cur_token + row_out.TOKEN):
                break

            cur_token += str(row_out.TOKEN)
            ne_tags.add(row_out['NE-TAG'])
            nel_ids.add(row_out['ID'])
            _, row_out = next(seq_out, (None, None))

        if row_gs.TOKEN != cur_token:
            tsv_out.append({'TOKEN': row_gs.TOKEN, 'NE-TAG': 'O', 'NE-EMB': 'O', 'ID': '-'})
            continue

        if len(ne_tags) == 1:
            ne_tag = ne_tags.pop()
        else:
            ne_tag = 'O'

        if len(nel_ids) == 1:
            nel_id = nel_ids.pop()
        else:
            nel_id = '-'

        if row_gs.url_id != url_id:

            if len(tsv_out) > 0:
                write_segment()

            url_id = row_gs.url_id

            with open(out_clef_file, 'a') as fw:
                fw.write(urls[row_gs.url_id])

        if row_gs.segment_id != segment_id:

            segment_id = row_gs.segment_id

            if len(tsv_out) > 0:
                write_segment()

        tsv_out.append({'TOKEN': cur_token, 'NE-TAG': ne_tag, 'NE-EMB': 'O', 'ID': nel_id})

    write_segment()


class SentenceStatTask:

    def __init__(self, entity_result, quantiles, rank_intervalls, min_pairs, max_pairs):

        self._entity_result = entity_result
        self._quantiles = quantiles
        self._rank_intervalls = rank_intervalls
        self._min_pairs = min_pairs
        self._max_pairs = max_pairs

    def __call__(self, *args, **kwargs):
        return features(dec=pd.read_json(json.dumps(self._entity_result['decision']), orient='split'),
                        cand=pd.read_json(json.dumps(self._entity_result['candidates']), orient='split'),
                        quantiles=self._quantiles, rank_intervalls=self._rank_intervalls,
                        min_pairs=self._min_pairs, max_pairs=self._max_pairs,
                        wikidata_gt=list(self._entity_result['gt'])[0])


@click.command()
@click.argument('tsv-file', type=click.Path(exists=True), required=True, nargs=1)
@click.argument('json-file', type=click.Path(exists=True), required=True, nargs=1)
@click.argument('clef-gs-file', type=click.Path(exists=True), required=True, nargs=1)
@click.argument('data-set-file', type=click.Path(), required=True, nargs=1)
@click.option('--min-pairs', type=int, default=10, help='default: 10.')
@click.option('--max-pairs', type=int, default=50, help='default: 50.')
@click.option('--processes', type=int, default=8, help='default: 8.')
def sentence_stat(tsv_file, json_file, clef_gs_file, data_set_file, min_pairs, max_pairs, processes):

    tsv = pd.read_csv(tsv_file, sep='\t', comment='#', quoting=3)
    tsv.loc[tsv.TOKEN.isnull(), 'TOKEN'] = ""

    tsv_gs = pd.read_csv(clef_gs_file, sep='\t', comment='#', quoting=3)
    tsv_gs.loc[tsv_gs.TOKEN.isnull(), 'TOKEN'] = ""

    with open(json_file, 'r') as fp_json:
        ned_result = json.load(fp_json)

    ned_result = add_ground_truth(ned_result, tsv, tsv_gs)

    results_with_gt = sum(['gt' in entity_result for _, entity_result in ned_result.items()])

    rank_intervalls = np.linspace(0.001, 0.1, 100)
    quantiles = np.linspace(0.1, 1, 10)

    def get_tasks():

        nonlocal rank_intervalls, quantiles

        for entity_id, entity_result in ned_result.items():

            if 'gt' not in entity_result:
                continue

            yield SentenceStatTask(entity_result, quantiles, rank_intervalls, min_pairs, max_pairs)

    progress = tqdm(prun(get_tasks(), processes=processes), total=results_with_gt)

    data = list()
    data_len = 0
    for data_part in progress:

        if data_part is None:
            continue

        data.append(data_part)
        data_len += len(data_part)

        progress.set_description("#data: {}".format(data_len))
        progress.refresh()

    data = pd.concat(data)

    data.to_pickle(data_set_file)


def add_ground_truth(ned_result, tsv, tsv_gs):
    ids = set()
    entity = ""
    entity_type = None

    def check_entity(tag):
        nonlocal entity, entity_type, ids

        if (entity != "") and ((tag == 'O') or tag.startswith('B-') or (tag[2:] != entity_type)):

            eid = entity + "-" + entity_type

            if eid in ned_result and 0 < len(ids) <= 1:

                if 'gt' in ned_result[eid]:
                    ned_result[eid]['gt'].union(ids)
                else:
                    ned_result[eid]['gt'] = ids

                try:
                    assert len(ned_result[eid]['gt']) <= 1
                except AssertionError:
                    import ipdb
                    ipdb.set_trace()

            ids = set()
            entity = ""
            entity_type = None

    seq_gs = tsv_gs.iterrows()
    _, row_gs = next(seq_gs, (None, None))
    cur_token = ""

    for rid, row in tsv.iterrows():

        if not str(row_gs.TOKEN).startswith(cur_token + str(row.TOKEN)):
            _, row_gs = next(seq_gs, (None, None))
            cur_token = ""

        assert row_gs is not None

        cur_token += str(row.TOKEN)
        # print("|{}||{}|".format(cur_token, row_gs.TOKEN))

        check_entity(row['NE-TAG'])

        if row['NE-TAG'] != 'O':
            entity_type = row['NE-TAG'][2:]

            entity += " " if entity != "" else ""

            entity += row['TOKEN']

            if row_gs.ID != 'NIL' and row_gs.ID != '_':
                ids.add(row_gs.ID)

    check_entity('O')

    _, row_gs = next(seq_gs, (None, None))
    assert row_gs is None

    return ned_result


teams = {

    'team1':
        {
            'name': 'ehrmama',
            'place': 'University of Amsterdam',
            'country': 'The Netherlands'
        }
    ,

    'team7':
        {
            'name': 'IRISA',
            'place': 'IRISA, Rennes',
            'country': 'France'
        }
    ,

    'team8':
        {
            'name': 'CISTeria',
            'place': 'Ludwig-Maximilians-Universität and Bayerische Staatsbibliothek München, Munich',
            'country': 'Germany'
        }
    ,

    'team10':
        {
            'name': 'L3i',
            'place': 'La Rochelle University, La Rochelle',
            'country': 'France'
        }
    ,

    'team11':
        {
            'name': 'NLP-UQAM',
            'place': 'Université du Quebec à Montréal, Montréal',
            'country': 'Quebec'
        }
    ,

    'team16':
        {
            'name': 'ERTIM',
            'place': 'Inalco, Paris',
            'country': 'France'
        }
    ,

    'team23':
        {
            'name': 'UPB',
            'place': 'Politehnica University of Bucharest, Bucarest',
            'country': 'Bulgaria'
        }
    ,

    'team28':
        {
            'name': 'SinNER',
            'place': 'INRIA and Paris-Sorbonne University, Paris',
            'country': 'France'
        }
    ,

    'team31':
        {
            'name': 'UvA.ILPS',
            'place': 'University of Amsterdam, Amsterdam',
            'country': 'The Netherlands'
        }
    ,

    'team33':
        {
            'name': 'SBB',
            'place': 'Berlin State Library, Berlin',
            'country': 'Germany'
        }
    ,

    'team37':
        {
            'name': 'Inria-DeLFT',
            'place': 'Almanach, Inria, Paris',
            'country': 'France'
        }
    ,

    'team39':
        {
            'name': 'LIMSI',
            'place': 'LIMSI, CNRS, Paris',
            'country': 'France'
        }
    ,

    'team40':
        {
            'name': 'Webis',
            'place': 'Webis group, Bauhaus University Weimar',
            'country': 'Germany'
        },
    'aidalight-baseline':
        {
            'name': 'aidalight-baseline'
        }
}


def read_HIPE_results():

    files = [f for f in glob.glob('*.tsv')]

    results = pd.concat([pd.read_csv(f, sep='\t') for f in files])
    results['team'] = results.System.str.extract('([^_]+).*')
    results['Task'] = results.System.str.extract('[^_]+[_](.*)')
    results['Lang'] = results.System.str.extract('^.+?_.+?_(.+?)_.+?$')

    results['Lang'] = results.Lang.str.upper()
    results.insert(0, 'team', results.pop('team'))
    results.insert(0, 'Lang', results.pop('Lang'))

    results['team'] = results.team.map(lambda i: teams[i]['name'] if i in teams else 'baseline')

    return results


def make_table_ner(results):

    sbb_results = results.loc[results.System.str.startswith('team33')]

    tmp = sbb_results.loc[(sbb_results.F1 > 0.1) & (sbb_results.Evaluation.str.startswith('NE-COARSE'))]

    tmp2 = pd.concat(
        [pd.concat([res.sort_values('F1', ascending=False).iloc[[0]] for _, res in results.loc[
            (results.Evaluation.str.startswith('NE-COARSE-LIT-micro-{}'.format(evaluation)))].groupby(
            'Lang')]).sort_values('F1', ascending=False) for evaluation in ['fuzzy', 'strict']])

    tmp = pd.concat([tmp, tmp2])

    tmp =\
        tmp.sort_values(['Lang', 'Evaluation', 'F1'], ascending=[True, True, False]).\
        drop(columns=['F1_std', 'P_std', 'R_std'])

    print(tmp[['Lang', 'team', 'Evaluation', 'Label', 'P', 'R', 'F1', 'TP', 'FP', 'FN']].to_latex(index=False))


def make_table_nel(results):

    sbb_results = results.loc[results.System.str.startswith('team33')]

    tmp = \
        sbb_results.loc[
            (sbb_results.System.str.startswith('team33_bundle2')) & (sbb_results.F1 > 0.1) &
            (sbb_results.Evaluation.str.startswith('NEL-LIT'))
        ].\
        sort_values(['System', 'Evaluation']).\
        drop(columns=['F1_std', 'P_std', 'R_std', 'Task'])

    print(tmp.to_latex(index=False))


def make_table_nel_only(results):

    sbb_results = results.loc[results.System.str.startswith('team33')]

    tmp =\
        sbb_results.loc[
            (sbb_results.System.str.startswith('team33_bundle5')) & (sbb_results.F1 > 0.1) &
            (sbb_results.Evaluation.str.startswith('NEL-LIT'))
        ].\
        sort_values(['System', 'Evaluation']).\
        drop(columns=['F1_std', 'P_std', 'R_std', 'Task'])

    print(tmp.to_latex(index=False))


def make_table_nel_comparison(results):
    lang = ['de', 'fr', 'en']

    tmp =\
        pd.concat(
            [pd.concat(
                [res.sort_values('P', ascending=False).iloc[[0]]
                 for _, res in results.loc[
                     results.System.str.contains('_bundle[1|2]_{}_'.format(lng)) &
                     results.Evaluation.str.startswith('NEL-LIT')].drop_duplicates().groupby('team')]
            ) for lng in lang]
        ).\
        drop(columns=['System', 'F1_std', 'P_std', 'R_std', 'TP', 'FP', 'FN', 'Task']).\
        sort_values(['Lang', 'P'], ascending=[False, False])

    print(tmp.to_latex(index=False))


def make_table_nel_only_comparison(results):

    lang = ['de', 'fr', 'en']

    tmp =\
        pd.concat(
            [pd.concat(
                [res.sort_values('P', ascending=False).iloc[[0]]
                 for _, res in results.loc[
                     results.System.str.contains('_bundle[5]_{}_'.format(lng)) &
                     results.Evaluation.str.startswith('NEL-LIT')].drop_duplicates().groupby('team')]
            ) for lng in lang]
        ).\
        drop(columns=['System', 'F1_std', 'P_std', 'R_std', 'TP', 'FP', 'FN', 'Task']).\
        sort_values(['Lang', 'P'], ascending=[False, False])

    print(tmp.to_latex(index=False))


@click.command()
@click.argument('entities-file', type=click.Path(exists=True), required=True, nargs=1)
@click.argument('wiki-db-file', type=click.Path(exists=True), required=True, nargs=1)
@click.argument('gt-file', type=click.Path(exists=True), required=True, nargs=1)
def compute_knb_coverage(entities_file, wiki_db_file, gt_file):

    dedata = pd.read_pickle(entities_file)

    with sqlite3.connect(wiki_db_file) as con:

        dewiki =\
            pd.read_sql(
                "select page_props.pp_value, page.page_id, page.page_title from page_props "
                "join page on page.page_id==page_props.pp_page where page.page_namespace == 0 and "
                "page_props.pp_propname == 'wikibase_item';", con)

    knb = dedata.merge(dewiki, left_index=True, right_on='page_title')

    test_data = pd.read_csv(gt_file, sep='\t', comment='#')

    entities_in_test_data =\
        test_data.loc[test_data['NEL-LIT'].str.len() > 1][['NEL-LIT']].drop_duplicates().reset_index(drop=True)

    with_representation = entities_in_test_data.merge(knb, left_on='NEL-LIT', right_on='pp_value')

    print("% of entities with representation: {}.".format(len(with_representation) / len(entities_in_test_data)))
