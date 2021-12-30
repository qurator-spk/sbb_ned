from ..embeddings.base import Embeddings
import numpy as np
import pandas as pd
import re
import unicodedata


class AnagramEmbeddings(Embeddings):

    def __init__(self, valid_chars=None, *args, **kwargs):

        super(AnagramEmbeddings, self).__init__(*args, **kwargs)

        if valid_chars is None:
            valid_chars = 'abcdefghijklmnopqrstuvwxyz'

        self._valid_chars = [valid_chars[i] for i in range(len(valid_chars))]

        self._filter = re.compile("[^{}]".format(valid_chars))

    def get(self, keys):

        for key in keys:
            key_nfkd = unicodedata.normalize('NFKD', key)

            key_ascii = key_nfkd.encode('ascii', 'ignore').decode()

            lower = key_ascii.lower()

            lower = self._filter.sub('', lower)

            counts = pd.DataFrame(self._valid_chars + [lower[i] for i in range(len(lower))], columns=["chars"]).\
                         chars.\
                         value_counts(sort=False).sort_index() - 1

            en = counts / np.sqrt(np.multiply(counts, counts).sum())

            yield key, en.values

    def config(self):

        return {'description': self.description()}

    def description(self):

        return "anagram-{}".format("".join(self._valid_chars))
