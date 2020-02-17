# from gensim.models.fasttext import FastText as FT_gensim
# from gensim.test.utils import datapath

from ..embeddings.base import Embeddings

from gensim.models.fasttext import load_facebook_vectors


class FastTextEmbeddings(Embeddings):

    def __init__(self, model_path, *args, **kwargs):

        super(FastTextEmbeddings, self).__init__(*args, **kwargs)

        self._path = model_path
        self._embeddings = None

    def get(self, keys):

        embeddings = self._emb()

        for key in keys:
            yield key, embeddings[key]

    def config(self):

        return {'description': self.description()}

    def description(self):

        return "fasttext-cc.de.300"

    def _emb(self):

        if self._embeddings is None:

            self._embeddings = load_facebook_vectors(self._path)

        return self._embeddings
