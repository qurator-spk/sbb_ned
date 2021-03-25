import qurator.sbb_ned.embeddings as emb


class FlairEmbeddings(emb.base.Embeddings):

    def __init__(self, forward, backward, use_tokenizer, *args, **kwargs):

        super(FlairEmbeddings, self).__init__(*args, **kwargs)

        self._forward = forward
        self._backward = backward
        self._use_tokenizer = use_tokenizer

        from flair.embeddings import FlairEmbeddings as FLEmbeddings
        from flair.embeddings import StackedEmbeddings

        self._embeddings = StackedEmbeddings([FLEmbeddings(forward), FLEmbeddings(backward)])

    def get(self, keys):

        from flair.data import Sentence

        sentences = [Sentence(key, use_tokenizer=self._use_tokenizer) for key in keys]

        # noinspection PyUnresolvedReferences
        self._embeddings.embed(sentences)

        for s_idx, sentence in enumerate(sentences):

            for t_idx, token in enumerate(sentence):

                yield token.text, token.embedding.cpu().numpy()

    def config(self):

        return {'description': self.description()}

    def description(self):

        return "flair-{}-{}".format(self._forward, self._backward)
