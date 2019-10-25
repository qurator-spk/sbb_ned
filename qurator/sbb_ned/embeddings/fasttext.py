import qurator.sbb_ned.embeddings as emb
# from gensim.models.fasttext import FastText as FT_gensim
# from gensim.test.utils import datapath
from gensim.models.fasttext import load_facebook_vectors


class FastTextEmbeddings(emb.base.Embeddings):

    def __init__(self, path, *args, **kwargs):

        super(FastTextEmbeddings, self).__init__(*args, **kwargs)

        self._embeddings = load_facebook_vectors(path)

        pass

    @staticmethod
    def dims():

        return 300

    def get(self, keys):

        for key in keys:
            yield key, self._embeddings[key]

    def config(self):

        return {'description': self.description(), 'dims': self.dims()}

    def description(self):

        return "fasttext-cc.de.300"
