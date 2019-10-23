from qurator.sbb_ned.index import Embeddings


class FlairEmbeddings(Embeddings):

    def __init__(self, forward, backward, use_tokenizer, *args, **kwargs):

        super(FlairEmbeddings, self).__init__(*args, **kwargs)

        self._forward = forward
        self._backward = backward
        self._use_tokenizer = use_tokenizer

        from flair.embeddings import FlairEmbeddings as FLEmbeddings
        from flair.embeddings import StackedEmbeddings

        self._embeddings = StackedEmbeddings([FLEmbeddings(forward), FLEmbeddings(backward)])

    @staticmethod
    def dims():
        return 4096

    def get(self, keys):

        from flair.embeddings import Sentence

        for key in keys:

            # print(key)

            sentence = Sentence(key, use_tokenizer=self._use_tokenizer)

            # noinspection PyUnresolvedReferences
            self._embeddings.embed(sentence)

            for token in sentence:

                yield token.text, token.embedding.cpu().numpy()

    def config(self):

        return {'description': self.description(), 'dims': self.dims()}

    def description(self):

        return "flair-{}-{}".format(self._forward, self._backward)
