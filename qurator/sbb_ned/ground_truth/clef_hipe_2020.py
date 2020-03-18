import click
import re
from io import StringIO
import pandas as pd


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

    df = df.rename(columns={'TOKEN_ID': 'No.', 'TOKEN': 'TOKEN', 'NE-COARSE-LIT': 'NE-TAG', 'NE-NESTED': 'NE-EMB',
                            'NEL-LIT': 'ID', })

    df.loc[~df['NE-TAG'].str[2:5].isin(entity_types), 'NE-TAG'] = 'O'
    df.loc[~df['NE-EMB'].str[2:5].isin(entity_types), 'NE-EMB'] = 'O'

    df = df[out_columns]

    pd.DataFrame([], columns=out_columns).to_csv(tsv_file, sep="\t", quoting=3, index=False)

    for (_, part), url in zip(df.groupby('url_id', sort=False, as_index=False), urls):

        with open(tsv_file, 'a') as f:
            f.write(url)

        part.to_csv(tsv_file, sep="\t", quoting=3, index=False, mode="a", header=False)
