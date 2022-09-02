import itertools
import os
import time
from functools import wraps

import numpy as np

from data_utils import read_corpus, convert_corpus_to_numeric, format_data
from test import evaluate_results
from train import train_model_2


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        print(f'func: {f.__name__} took:{te - ts} sec')
        return result

    return wrap


def find_match(words_dict, t_e_f, french_word):
    f_idx = words_dict['french'][french_word]
    e_idx = np.argmax(t_e_f[:, f_idx])
    return list(words_dict['english'].keys())[e_idx]


if __name__ == "__main__":
    mini = False
    if mini:
        mini_str = '_mini'
    else:
        mini_str = ''
    english_corpus_path = r'./word-alignment/data/hansards.e'
    french_corpus_path = r'./word-alignment/data/hansards.f'
    output_test = r'results_model2/results_model_2.txt'
    t_e_f_path = r'results_model2/tef_iter14_model2.npy'

    n_iter = 1

    corpus, sentence_len = read_corpus(english_corpus_path, french_corpus_path)
    # a set with all of the words (they will be referred by index)
    # clean_corpus(corpus)
    words_dict, n_words = format_data(corpus)

    # implement the arrays as 2d numpy array
    total_french_words = len(list(itertools.chain.from_iterable(corpus['french'])))
    t_e_f = np.zeros((n_words['english'], n_words['french'])) + 1 / float(total_french_words)
    max_len = {'french': max(sentence_len['french']), 'english': max(sentence_len['english'])}
    q = np.zeros(
        (max_len['french'], max_len['english'], max_len['french'], max_len['english']),
        dtype=np.float32)  # q[f_pos, e_pos, f_len, e_len]
    for lf in range(max_len['french']):
        q[:, :, lf, :] = 1 / (lf + 1)

    # convert to numbers
    numeric_corpus = convert_corpus_to_numeric(corpus, words_dict)

    if os.path.exists(t_e_f_path):
        with open(t_e_f_path, 'rb') as f:
            t_e_f = np.load(f)

    # train the model
    t_e_f, q = train_model_2(t_e_f, q, n_iter, numeric_corpus, max_len, n_words)

    # tef_rows = t_e_f.shape[0]
    with open(t_e_f_path, 'wb') as f:
        np.save(f, t_e_f)

    evaluate_results(numeric_corpus, t_e_f, output_test)
    print('done!')
