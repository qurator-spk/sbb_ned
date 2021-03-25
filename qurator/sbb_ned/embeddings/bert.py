from ..embeddings.base import Embeddings

from flair.data import Sentence


class BertEmbeddings(Embeddings):

    def __init__(self, model_path,
                 layers="-1, -2, -3, -4", pooling_operation='first', use_scalar_mix=True, no_cuda=False, *args, **kwargs):

        super(BertEmbeddings, self).__init__(*args, **kwargs)

        self._path = model_path
        self._embeddings = None
        self._layers = layers
        self._pooling_operation = pooling_operation
        self._use_scalar_mix = use_scalar_mix
        self._no_cuda = no_cuda

    def get(self, keys):

        if self._embeddings is None:

            if self._no_cuda:
                import flair
                import torch
                flair.device = torch.device('cpu')

            from .flair_bert import BertEmbeddings

            self._embeddings = BertEmbeddings(bert_model_or_path=self._path,
                                              layers=self._layers,
                                              pooling_operation=self._pooling_operation,
                                              use_scalar_mix=self._use_scalar_mix)

        sentences = [Sentence(key) for key in keys]

        # noinspection PyUnresolvedReferences
        self._embeddings.embed(sentences)

        for s_idx, sentence in enumerate(sentences):

            for t_idx, token in enumerate(sentence):

                emb = token.embedding.cpu().numpy()
                tok = str(token)

                yield tok, emb

                del token

            del sentence

    def config(self):

        return {'description': self.description()}

    def description(self):

        layer_str = self._layers
        layer_str = layer_str.replace(' ', '')
        layer_str = layer_str.replace(',', '_')

        return "bert-layers_{}-pooling_{}-scalarmix_{}".format(layer_str, self._pooling_operation, self._use_scalar_mix)
