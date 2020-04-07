import click
import pandas as pd
import logging

logger = logging.getLogger(__name__)


@click.command()
@click.argument('text-file', type=click.Path(exists=True), required=True, nargs=1)
@click.argument('table-file', type=click.Path(exists=False), required=True, nargs=1)
def extract_normalization_table(text_file, table_file):

    markers = ['MUFI', 'Unicode', 'Private1', 'Private2']

    lines = [w for l in open(text_file) for w in l.strip().split(" ")]

    df = pd.DataFrame(lines, columns=['text'])

    df = df.loc[df.text.str.len() > 0].reset_index(drop=True)

    def conv(v, base):
        try:
            return int(v, base)
        except ValueError:
            return -1
        except AssertionError:
            return -1

    df['isdecimal'] = df.text.str.isdecimal()
    df['decimal'] = df.text.apply(lambda s: conv(s, 10))
    df['hexa'] = df.text.apply(lambda s: conv(s, 16))
    df['ishexa'] = df.hexa != -1
    df['ismarker'] = df.text.isin(markers)

    table = list()

    def find_decimal_before(start):

        for p in range(start, 0, -1):
            if df.loc[p].isdecimal:
                return df.loc[p].decimal, p

        return None, None

    def find_decimal_after(start):

        for p in range(start, len(df)):
            if df.loc[p].isdecimal:
                return df.loc[p].decimal, p

        return None, None

    def find_hexa_before(start):

        for p in range(start, 0, -1):
            if df.loc[p].ishexa:
                return df.loc[p].text, df.loc[p].hexa, p

        return None, None, None

    for pos in df.loc[df.ismarker].index:

        origin = df.loc[pos].text
        base = df.loc[pos + 1].text

        if len(base) > 1:
            logger.debug('len(base) > 1')
            continue

        decimal_aft, _ = find_decimal_after(pos + 1)

        combining_character = ''
        if df.loc[pos + 2].ishexa:
            combining_character = df.loc[pos + 2].text

        if decimal_aft == ord(base):
            base = ''
            combining_character = ''

        decimal, decimal_pos = find_decimal_before(pos-1)

        if decimal is None:
            logger.debug('decimal is None.')
            continue

        hexstr, hexa, hex_pos = find_hexa_before(decimal_pos-1)

        if hexa != decimal:
            logger.debug('hexa != decimal')
            continue

        unicode = chr(decimal)

        description = " ".join(df.loc[decimal_pos+1:pos-1].text.astype(str).tolist())

        table.append((unicode, hexstr, decimal, description, origin, base, combining_character))

    table = pd.DataFrame(table, columns=['unicode', 'hex', 'decimal', 'description', 'origin', 'base',
                                         'combining_character'])
    table.to_pickle(table_file)
