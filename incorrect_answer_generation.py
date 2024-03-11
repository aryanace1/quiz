import nltk
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import wordnet
import gensim
import gensim.downloader as api
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize, word_tokenize
import random
import numpy as np

class IncorrectAnswerGenerator:
    ''' This class contains the methods
    for generating the incorrect answers
    given an answer
    '''

    def __init__(self, document):
        # model required to fetch similar words
        self.model = api.load("glove-wiki-gigaword-100")
        self.all_words = []
        for sent in sent_tokenize(document):
            self.all_words.extend(word_tokenize(sent))
        self.all_words = list(set(self.all_words))
        self.pos_tagger = nltk.pos_tag

    def get_similar_words_same_pos(self, word, pos_tag):
        similar_words = []
        try:
            for sim_word, _ in self.model.most_similar(word, topn=100):
                sim_word_pos = self.pos_tagger([sim_word])[0][1]
                if sim_word_pos == pos_tag and sim_word != word:
                    similar_words.append(sim_word)
                if len(similar_words) >= 3:
                    break
        except KeyError:
            pass  # Word not in vocabulary
        return similar_words

    def get_all_options_dict(self, answer, num_options):
        ''' This method returns a dict
        of 'num_options' options out of
        which one is correct and is the answer
        '''
        options_dict = dict()
        similar_words = self.get_similar_words_same_pos(answer, self.pos_tagger([answer])[0][1])
        for i in range(1, num_options + 1):
            if i == num_options:
                options_dict[i] = answer
            else:
                options_dict[i] = random.choice(similar_words) if similar_words else answer

        return options_dict


# Example usage:
# document = "This is a sample document. It contains some words."
# generator = IncorrectAnswerGenerator(document)

# correct_answer = "sample"
# num_options = 4
# options_dict = generator.get_all_options_dict(correct_answer, num_options)
# print("Correct Answer:", correct_answer)
# print("Options:")
# for option, answer in options_dict.items():
#     print(f"{option}: {answer}")


# ''' This module contains the class
# for generating incorrect alternative
# answers for a given answer
# '''
# import gensim
# import gensim.downloader as api
# from gensim.models import Word2Vec
# from nltk.tokenize import sent_tokenize, word_tokenize
# import random
# import numpy as np


# class IncorrectAnswerGenerator:
#     ''' This class contains the methods
#     for generating the incorrect answers
#     given an answer
#     '''

#     def __init__(self, document):
#         # model required to fetch similar words
#         self.model = api.load("glove-wiki-gigaword-100")
#         self.all_words = []
#         for sent in sent_tokenize(document):
#             self.all_words.extend(word_tokenize(sent))
#         self.all_words = list(set(self.all_words))

#     def get_all_options_dict(self, answer, num_options):
#         ''' This method returns a dict
#         of 'num_options' options out of
#         which one is correct and is the answer
#         '''
#         options_dict = dict()
#         try:
#             similar_words = self.model.similar_by_word(answer, topn=15)[::-1]

#             for i in range(1, num_options + 1):
#                 options_dict[i] = similar_words[i - 1][0]

#         except BaseException:
#             self.all_sim = []
#             for word in self.all_words:
#                 if word not in answer:
#                     try:
#                         self.all_sim.append(
#                             (self.model.similarity(answer, word), word))
#                     except BaseException:
#                         self.all_sim.append(
#                             (0.0, word))
#                 else:
#                     self.all_sim.append((-1.0, word))

#             self.all_sim.sort(reverse=True)

#             for i in range(1, num_options + 1):
#                 options_dict[i] = self.all_sim[i - 1][1]

#         replacement_idx = random.randint(1, num_options)

#         options_dict[replacement_idx] = answer

#         return options_dict
