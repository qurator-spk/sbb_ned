# from gensim.models.fasttext import FastText as FT_gensim
# from gensim.test.utils import datapath

from ..embeddings.base import Embeddings

from gensim.models.fasttext import load_facebook_vectors


class FastTextEmbeddings(Embeddings):

    def __init__(self, path, *args, **kwargs):

        super(FastTextEmbeddings, self).__init__(*args, **kwargs)

        self._path = path
        self._embeddings = None


    @staticmethod
    def dims():

        return 300

    def get(self, keys):

        if self._embeddings is None:

            self._embeddings = load_facebook_vectors(self._path)

        for key in keys:
            yield key, self._embeddings[key]

    def config(self):

        return {'description': self.description(), 'dims': self.dims()}

    def description(self):

        return "fasttext-cc.de.300"
