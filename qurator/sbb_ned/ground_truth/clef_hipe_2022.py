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
from qurator.utils.tsv import read_tsv
from ..models.decider import features

logger = logging.getLogger(__name__)


def read_clef(clef_file):
    with open(clef_file, 'r') as f:

        segments = []
        tsv_part = []
        header = None
        contexts = []

        def make_segement():

            nonlocal segments, tsv_part

            if len(tsv_part) == 0:
                return

            segment = None
            # noinspection PyBroadException
            try:
                segment = pd.read_csv(StringIO(header + "".join(tsv_part)), sep='\t', comment='#', quoting=3)
            except Exception as e:
                print(e)
                import ipdb
                ipdb.set_trace()

            segment = segment.reset_index().rename(columns={'index': 'TOKEN_ID'})

            segment['url_id'] = len(segments)

            tsv_part = []

            token_id = []
            counter = 0
            for misc in segment.MISC.astype(str).to_list():

                token_id.append(counter)

                if re.match(r'.*EndOfSentence.*', misc):
                    counter = 0
                else:
                    counter += 1

            segment['TOKEN_ID'] = token_id

            segments.append(segment)

        context = dict()

        for line in tqdm(f):

            if header is None:
                header = "\t".join(line.split()) + '\n'
                continue

            m = re.match(r'#\s+(.*)\s+=\s+(.*)', line)

            if m:
                if len(tsv_part) > 0:
                    make_segement()
                    contexts.append(context)

                    context = dict()

                context[m.group(1)] = m.group(2)
            else:
                tsv_part.append(line)

        make_segement()
        contexts.append(context)

        return contexts, pd.concat(segments).reset_index(drop=True)


@click.command()
@click.argument('clef-file', type=click.Path(exists=True), required=True, nargs=1)
@click.argument('tsv-file', type=click.Path(), required=True, nargs=1)
def cli_clef2tsv(clef_file, tsv_file):
    clef2tsv(clef_file, tsv_file)


def clef2tsv(clef_file, tsv_file):
    out_columns = ['No.', 'TOKEN', 'NE-TAG', 'NE-EMB', 'ID', 'url_id', 'left', 'right', 'top', 'bottom']

    entity_types = ['PER', 'LOC', 'ORG', 'O']

    contexts, df = read_clef(clef_file)

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

    pd.DataFrame([], columns=df.columns).to_csv(tsv_file, sep="\t", quoting=3, index=False)

    for (_, part), context in zip(df.groupby('url_id', sort=False, as_index=False), contexts):
        with open(tsv_file, 'a') as f:
            f.write('#__CONTEXT__:{}\n'.format(json.dumps(context)))

        part.to_csv(tsv_file, sep="\t", quoting=3, index=False, mode="a", header=False)


@click.command()
@click.argument('tsv-file', type=click.Path(exists=True), required=True, nargs=1)
@click.argument('clef-gs-file', type=click.Path(), required=True, nargs=1)
@click.argument('out-clef-file', type=click.Path(), required=True, nargs=1)
def cli_tsv2clef(tsv_file, clef_gs_file, out_clef_file):
    tsv2clef(tsv_file, clef_gs_file, out_clef_file)


def tsv2clef(tsv_file, clef_gs_file, out_clef_file):
    out_columns = ['TOKEN', 'NE-COARSE-LIT', 'NE-COARSE-METO', 'NE-FINE-LIT', 'NE-FINE-METO', 'NE-FINE-COMP',
                   'NE-NESTED', 'NEL-LIT', 'NEL-METO', 'MISC']

    tsv = pd.read_csv(tsv_file, sep='\t', comment='#', quoting=3)
    tsv.loc[tsv.TOKEN.isnull(), 'TOKEN'] = ""

    contexts, tsv_gs = read_clef(clef_gs_file)
    tsv_gs['TOKEN'] = tsv_gs.TOKEN.astype(str)

    tsv_out = []

    pd.DataFrame([], columns=out_columns).to_csv(out_clef_file, sep="\t", quoting=3, index=False)

    def write_segment():

        nonlocal tsv_out, out_columns

        tsv_out = pd.DataFrame(tsv_out). \
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
                fw.write(contexts[row_gs.url_id])

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
@click.option('--context-split', type=bool, is_flag=True, help="Perform extraction for different contexts.")
def sentence_stat(tsv_file, json_file, clef_gs_file, data_set_file, min_pairs, max_pairs, processes, context_split):

    if not context_split:
        tsv = pd.read_csv(tsv_file, sep='\t', comment='#', quoting=3)
        tsv.loc[tsv.TOKEN.isnull(), 'TOKEN'] = ""

        tsv_gs = pd.read_csv(clef_gs_file, sep='\t', comment='#', quoting=3)
        tsv_gs.loc[tsv_gs.TOKEN.isnull(), 'TOKEN'] = ""

        with open(json_file, 'r') as fp_json:
            ned_result = json.load(fp_json)

        data = _sentence_stat(ned_result, tsv, tsv_gs, min_pairs, max_pairs, processes)

    else:
        tsv, urls, contexts = read_tsv(tsv_file)

        tsv = [(url_id, part) for url_id, part in tsv.groupby("url_id", as_index=False)]

        tsv_gs, urls_gs, contexts_gs = read_tsv(clef_gs_file)

        tsv_gs = [(url_id, part) for url_id, part in tsv_gs.groupby("url_id", as_index=False)]

        assert(len(tsv) == len(tsv_gs))

        assert(len(tsv) == len(contexts))
        assert (len(tsv_gs) == len(contexts_gs))

        with open(json_file, 'r') as fp_json:
            ned_result = json.load(fp_json)

        assert(len(ned_result) == len(tsv))
        assert (len(ned_result) == len(tsv_gs))

        data = []
        for (_, tsv_cur), (_, tsv_gs_cur), context_cur, context_gs_cur, ned_result_cur in \
                zip(tsv, tsv_gs, contexts, contexts_gs, ned_result):

            data_cur = _sentence_stat(ned_result_cur, tsv_cur, tsv_gs_cur, min_pairs, max_pairs, processes)

            if len(data_cur) < 1:
                continue

            data.append(data_cur)

        data = pd.concat(data).reset_index(drop=True)

    data.to_pickle(data_set_file)


def _sentence_stat(ned_result, tsv, tsv_gs, min_pairs, max_pairs, processes):
    ned_result = add_ground_truth(ned_result, tsv, tsv_gs)

    applicable_results = sum(['gt' in entity_result and 'decision' in entity_result
                              for _, entity_result in ned_result.items()])

    rank_intervalls = np.linspace(0.001, 0.1, 100)
    quantiles = np.linspace(0.1, 1, 10)

    def get_tasks():

        nonlocal rank_intervalls, quantiles

        for entity_id, entity_result in ned_result.items():

            if 'gt' not in entity_result:
                continue

            if 'decision' not in entity_result:
                continue

            yield SentenceStatTask(entity_result, quantiles, rank_intervalls, min_pairs, max_pairs)

    progress = tqdm(prun(get_tasks(), processes=processes), total=applicable_results)

    data = list()
    data_len = 0
    for data_part in progress:

        if data_part is None:
            continue

        data.append(data_part)
        data_len += len(data_part)

        progress.set_description("#data: {}".format(data_len))
        progress.refresh()

    if len(data) < 1:
        return pd.DataFrame()

    data = pd.concat(data)

    return data


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

    tmp = \
        tmp.sort_values(['Lang', 'Evaluation', 'F1'], ascending=[True, True, False]). \
            drop(columns=['F1_std', 'P_std', 'R_std'])

    print(tmp[['Lang', 'team', 'Evaluation', 'Label', 'P', 'R', 'F1']].to_latex(index=False))


def make_table_nel(results):
    tmp = results.loc[results.System.str.startswith('team33_bundle2') &
                      results.Evaluation.str.startswith('NEL-LIT-micro-fuzzy')].drop_duplicates()

    tmp = tmp. \
        sort_values(['System', 'Evaluation']). \
        drop(columns=['TP', 'FN', 'FP', 'F1_std', 'P_std', 'R_std', 'Task', 'System'])

    print(tmp.to_latex(index=False))


def make_table_nel_only(results):
    tmp = results.loc[results.System.str.startswith('team33_bundle5') &
                      results.Evaluation.str.startswith('NEL-LIT-micro-fuzzy')].drop_duplicates()

    tmp = tmp. \
        sort_values(['System', 'Evaluation']). \
        drop(columns=['TP', 'FN', 'FP', 'F1_std', 'P_std', 'R_std', 'Task', 'System'])

    print(tmp.to_latex(index=False))


def make_table_nel_comparison(results):
    lang = ['de', 'fr', 'en']

    tmp = \
        pd.concat(
            [pd.concat(
                [res.sort_values('P', ascending=False).iloc[[0]]
                 for _, res in results.loc[
                     results.System.str.contains('_bundle[1|2]_{}_'.format(lng)) &

                     results.Evaluation.str.startswith('NEL-LIT-micro-fuzzy-relaxed-@5')].drop_duplicates().
                     groupby('team')]
            ) for lng in lang]
        ). \
            drop(columns=['System', 'F1_std', 'P_std', 'R_std', 'TP', 'FP', 'FN', 'Task']). \
            sort_values(['Lang', 'P'], ascending=[False, False])

    print(tmp.to_latex(index=False))


def make_table_nel_only_comparison(results):
    lang = ['de', 'fr', 'en']

    tmp = \
        pd.concat(
            [pd.concat(
                [res.sort_values('P', ascending=False).iloc[[0]]
                 for _, res in results.loc[
                     results.System.str.contains('_bundle[5]_{}_'.format(lng)) &
                     results.Evaluation.str.startswith('NEL-LIT-micro-fuzzy-relaxed-@5')].drop_duplicates().
                     groupby('team')]
            ) for lng in lang]
        ). \
            drop(columns=['System', 'F1_std', 'P_std', 'R_std', 'TP', 'FP', 'FN', 'Task']). \
            sort_values(['Lang', 'P'], ascending=[False, False])

    print(tmp.to_latex(index=False))


@click.command()
@click.argument('entities-file', type=click.Path(exists=True), required=True, nargs=1)
@click.argument('gt-file', type=click.Path(exists=True), required=True, nargs=1)
def compute_knb_coverage(entities_file, gt_file):
    knb = pd.read_pickle(entities_file)

    test_data = pd.read_csv(gt_file, sep='\t', comment='#')

    entities_in_test_data = \
        test_data.loc[(test_data['NEL-LIT'].str.len() > 1) & (test_data['NEL-LIT'] != 'NIL')][
            ['NEL-LIT']].drop_duplicates().reset_index(drop=True)

    with_representation = entities_in_test_data.merge(knb, left_on='NEL-LIT', right_on='QID')

    print("% of entities with representation: {}.".format(len(with_representation) / len(entities_in_test_data)))


def compute_nil_fraction(gt_file):
    vc = pd.read_csv(gt_file, sep='\t', comment='#')['NEL-LIT'].value_counts()

    print(vc['NIL'] / (vc.sum() - vc['NIL'] - vc['_']))

