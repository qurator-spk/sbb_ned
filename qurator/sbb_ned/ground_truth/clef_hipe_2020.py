import click
import re
from io import StringIO
import pandas as pd


@click.command()
@click.argument('clef-file', type=click.Path(exists=True), required=True, nargs=1)
@click.argument('tsv-file', type=click.Path(), required=True, nargs=1)
def clef2tsv(clef_file, tsv_file):

    with open(clef_file, 'r') as f:

        parts = []
        part = []
        header = None
        urls = []

        for line in f:

            if header is None:
                header = "\t".join(line.split()) + '\n'
                continue

            if not line.startswith('#'):
                part.append(line)

            if re.match(r'#\s+document_id\s+=.*', line) and len(part) > 0:

                urls.append(line)

                df = pd.read_csv(StringIO(header + "".join(part)), sep='\t', comment='#')

                df = df.reset_index().rename(columns={'index': 'TOKEN_ID'})

                df['url_id'] = len(parts)

                parts.append(df)

                part = []

    df = pd.concat(parts).reset_index(drop=True)

    df['NE-COARSE-LIT'] = df['NE-COARSE-LIT'].str[0:5].str.upper()

    df['NE-NESTED'] = df['NE-NESTED'].str[0:5].str.upper()

    df['left'] = 0
    df['right'] = 0
    df['top'] = 0
    df['bottom'] = 0

    out_columns = ['No.', 'TOKEN', 'NE-TAG', 'NE-EMB', 'ID', 'url_id', 'left', 'right', 'top', 'bottom']

    df = df.rename(columns={'TOKEN_ID': 'No.', 'TOKEN': 'TOKEN', 'NE-COARSE-LIT': 'NE-TAG', 'NE-NESTED': 'NE-EMB',
                            'NEL-LIT': 'ID', })

    entity_types = ['PER', 'LOC', 'ORG', 'O']

    df.loc[~df['NE-TAG'].str[2:5].isin(entity_types), 'NE-TAG'] = 'O'
    df.loc[~df['NE-EMB'].str[2:5].isin(entity_types), 'NE-EMB'] = 'O'

    df = df[out_columns]

    pd.DataFrame([], columns=out_columns).to_csv(tsv_file, sep="\t", quoting=3, index=False)

    for (id, part), url in zip(df.groupby('url_id', sort=False, as_index=False), urls):

        with open(tsv_file, 'a') as f:
            f.write(url)

        part.to_csv(tsv_file, sep="\t", quoting=3, index=False, mode="a", header=False)
