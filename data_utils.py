import itertools
import time
from functools import wraps

import numpy as np


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        print(f'func: {f.__name__} took:{te - ts} sec')
        return result

    return wrap


def clean_corpus(corpus, languages=None):
    """
    Cleans the input - removes unwanted characters, lowercase etc...
    :param corpus: a list of sentences each one is a list of words
    :param languages: the required langages
    :return: normalised corpus
    """
    if languages is None:
        languages = ['english', 'french']
    for language in languages:
        for sentence in corpus[language]:
            remove_words = []
            for word in sentence:
                # if 'A' > min(word) or 'z' < max(word):
                if word == ',' or word == '.':
                    remove_words.append(word)
            for word in remove_words:
                sentence.remove(word)


def read_corpus(english_corpus_path, french_corpus_path):
    corpus = {}
    sentence_len = {}
    with open(english_corpus_path, encoding='utf-8') as f:
        corpus['english'] = [x.strip().split(' ') for x in f.readlines()]
        sentence_len['english'] = [len(x) for x in corpus['english']]
    with open(french_corpus_path, encoding='utf-8') as f:
        corpus['french'] = [x.strip().split(' ') for x in f.readlines()]
        sentence_len['french'] = [len(x) for x in corpus['french']]
    return corpus, sentence_len


@timing
def convert_corpus_to_numeric(corpus, words_dict, languages=None):
    if languages is None:
        languages = ['french', 'english']
    numeric_corpus = {}
    for language in languages:
        numeric_corpus[language] = []
        for sentence in corpus[language]:
            numeric_sentence = np.zeros(len(sentence))
            for idx, word in enumerate(sentence):
                word_idx = words_dict[language][word]
                numeric_sentence[idx] = word_idx
            numeric_corpus[language].append(np.int16(numeric_sentence))
    return numeric_corpus


def format_data(corpus, languages=None):
    if languages is None:
        languages = ['french', 'english']
    words_dict = {}
    n_words = {}
    for language in languages:
        words_dict[language] = {}
        words = tuple(sorted(set(itertools.chain.from_iterable(corpus[language]))))
        for idx, word in enumerate(words):
            words_dict[language][word] = idx
        n_words[language] = len(words)

    return words_dict, n_words
