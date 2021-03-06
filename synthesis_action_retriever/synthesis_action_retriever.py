# coding=utf-8
import os
import numpy as np

from gensim.models import Word2Vec
from tensorflow import keras

from synthesis_action_retriever.utils import replace_token_upd, make_spacy_tokens

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class SynthActionRetriever:
    def __init__(self,
                 embedding_model="path-to-model",
                 extractor_model="path-to-model"
                 ):

        print("Synthesis Action Retriever v1.0.0")

        self.__embeddings = Word2Vec.load(embedding_model)
        self.__model = keras.models.load_model(extractor_model)

        # declare operation types
        self.__num2action = {
                                0: "",
                                1: 'Starting',
                                2: 'Mixing',
                                3: 'Purification',
                                4: 'Heating',
                                5: 'Shaping',
                                6: 'Cooling',
                                7: 'Reaction'
                            }
        self._num_classes = 8
        self.__input_word_dim = self.__embeddings.trainables.layer1_size

        print("Done initialization.")

    def __get_embeddings(self, word):
        """
            returns Word2Vec embedding from gensim for single word token
        :param word: single word token
        :return: Word2Vec embedding
        """

        if word in ["<start>", "<end>"]:
            return np.zeros(self.__input_word_dim, dtype=float)

        if word in self.__embeddings.wv.vocab:
            return self.__embeddings.wv.__getitem__(word)
        else:
            return self.__embeddings.wv.__getitem__("<unk>")

    def __get_sentence_vector(self, spacy_tokens):
        """
            vectorizes input sentence tokens using loaded spacy nlp model
        :param sentence_toks: tokens from given sentence
        :return: vectorized sentence using Word2Vec embeddings
        """
        input_sentences_data = np.zeros((1, len(spacy_tokens) + 2, self.__input_word_dim), dtype='float32')
        input_sentences_data[0, 0] = self.__get_embeddings("<start>")
        for t, word in enumerate(spacy_tokens):
            embed_vec = self.__get_embeddings(replace_token_upd(word, mode=""))
            input_sentences_data[0, t] = embed_vec
        input_sentences_data[0, -1] = self.__get_embeddings("<end>")

        return input_sentences_data

    def get_action_labels(self, sentence="", spacy_tokens=None):
        """
            finds actions among sentence tokens and classifies them according to its type
        :param sentence: can be text string
        :param spacy_tokens: list of text tokens or spacy processed tokens
        :return: list of operations
        """
        if not spacy_tokens:
            spacy_tokens = make_spacy_tokens(sentence)
        sentence_vector = self.__get_sentence_vector(spacy_tokens)
        prediction = self.__model.predict(sentence_vector)[0]
        tags_predicted = [self.__num2action[np.argmax(v)] for v in prediction][0:len(spacy_tokens)]

        return tags_predicted
