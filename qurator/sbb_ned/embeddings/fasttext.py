from qurator.sbb_ned.index import Embeddings
# from gensim.models.fasttext import FastText as FT_gensim
# from gensim.test.utils import datapath
from gensim.models.fasttext import load_facebook_vectors


class FastTextEmbeddings(Embeddings):

    def __init__(self, path, *args, **kwargs):

        super(FastTextEmbeddings, self).__init__(*args, **kwargs)

        self._embeddings = load_facebook_vectors(path)

        pass

    def dims(self):

        return 300

    def get(self, key):

        return self._embeddings[key]

    def description(self):

        return "fasttext-cc.de.300"
