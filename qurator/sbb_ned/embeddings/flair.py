from qurator.sbb_ned.index import Embeddings

from flair.embeddings import FlairEmbeddings as FLEmbeddings
from flair.embeddings import StackedEmbeddings, Sentence
# import flair
# import torch

# flair.device = torch.device('cpu')


class FlairEmbeddings(Embeddings):

    def __init__(self, forward, backward, *args, **kwargs):

        super(FlairEmbeddings, self).__init__(*args, **kwargs)

        self._forward = forward
        self._backward = backward
        #self._flair_embedding_forward =
        #self._flair_embedding_backward =

        self._embeddings = StackedEmbeddings([FLEmbeddings(forward), FLEmbeddings(backward))

    def dims(self):

        return 4096

    def get(self, keys):

        for key in keys:
            sentence = Sentence(key)

            # noinspection PyUnresolvedReferences
            self._embeddings.embed(sentence)

            for token in sentence:

                vals = token.embedding.cpu().numpy()

                yield token.text, vals

                del vals
                del token

            del sentence

    def config(self):

        return {'description': self.description(), 'dims': self.dims()}

    def description(self):

        return "flair-{}-{}".format(self._forward, self._backward)
