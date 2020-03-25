import click
import re
from io import StringIO
import pandas as pd
import unicodedata
from tqdm import tqdm


def read_clef(clef_file):

    with open(clef_file, 'r') as f:

        parts = []
        part = []
        header = None
        urls = []

        def make_part():

            nonlocal part, parts

            if len(part) == 0:
                return

            tmp = pd.read_csv(StringIO(header + "".join(part)), sep='\t', comment='#')

            tmp = tmp.reset_index().rename(columns={'index': 'TOKEN_ID'})

            tmp['url_id'] = len(parts)

            parts.append(tmp)

            part = []

        for line in f:

            if header is None:
                header = "\t".join(line.split()) + '\n'
                continue

            if not line.startswith('#'):
                part.append(line)

            if re.match(r'#\s+document_id\s+=.*', line):

                urls.append(line)

                make_part()

        make_part()

        return urls, pd.concat(parts).reset_index(drop=True)


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
        df.loc[idx, 'TOKEN'] = "".join([c if unicodedata.category(c) != 'Cc' else '' for c in row.TOKEN])

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

    tsv = pd.read_csv(tsv_file, sep='\t', comment='#', quoting=3)

    tsv_gs = pd.read_csv(clef_gs_file, sep='\t', comment='#')

    tsv.loc[tsv.TOKEN.isnull(), 'TOKEN'] = ""

    seq_out = tsv.iterrows()

    _, row_out = next(seq_out, (None, None))
    tsv_out = list()
    for _, row_gs in tsv_gs.iterrows():

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

        tsv_out.append({'TOKEN': cur_token, 'NE-TAG': ne_tag, 'NE-EMB': 'O', 'ID': nel_id})

    tsv_out = pd.DataFrame(tsv_out).rename(columns={'NE-TAG': 'NE-COARSE-LIT', 'NE-EMB': 'NE-NESTED', 'ID': 'NEL-LIT'})

    tsv_out['NE-COARSE-LIT'] = tsv_out['NE-COARSE-LIT'].str.replace('-PER', '-pers')
    tsv_out['NE-COARSE-LIT'] = tsv_out['NE-COARSE-LIT'].str.replace('-LOC', '-loc')
    tsv_out['NE-COARSE-LIT'] = tsv_out['NE-COARSE-LIT'].str.replace('-ORG', '-org')

    tsv_out['NE-NESTED'] = tsv_out['NE-NESTED'].str.replace('-PER', '-pers')
    tsv_out['NE-NESTED'] = tsv_out['NE-NESTED'].str.replace('-LOC', '-loc')
    tsv_out['NE-NESTED'] = tsv_out['NE-NESTED'].str.replace('-ORG', '-org')

    tsv_out['NE-COARSE-METO'] = 'O'
    tsv_out['NE-FINE-LIT'] = 'O'
    tsv_out['NE-FINE-METO'] = 'O'
    tsv_out['NE-FINE-COMP'] = 'O'
    tsv_out['NEL-METO'] = '-'
    tsv_out['MISC'] = '-'

    out_columns = ['TOKEN', 'NE-COARSE-LIT', 'NE-COARSE-METO', 'NE-FINE-LIT', 'NE-FINE-METO', 'NE-FINE-COMP',
                   'NE-NESTED', 'NEL-LIT', 'NEL-METO', 'MISC']

    tsv_out = tsv_out[out_columns]

    tsv_out.to_csv(out_clef_file, sep="\t", index=False)
