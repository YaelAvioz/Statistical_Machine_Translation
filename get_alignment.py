import itertools
import os
import time
from functools import wraps

import numpy as np
from tqdm import tqdm

from data_utils import read_corpus, convert_corpus_to_numeric, format_data


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        print(f'func: {f.__name__} took:{te - ts} sec')
        return result

    return wrap


@timing
def evaluate_sentence(numeric_corpus, total_f, counter):
    """
    calculates the statistics of all of the sentences
    :param numeric_corpus: the corpus with the word indices
    :param total_f: the matrix with the f stats
    :param counter: the
    :return: updated total_f and counter
    """
    for e_sentence, f_sentence in tqdm(zip(numeric_corpus['english'], numeric_corpus['french'])):
        s_total_e = np.zeros(len(e_sentence))
        for sentence_idx, e_idx in enumerate(e_sentence):
            for f_idx in f_sentence:
                s_total_e[sentence_idx] += t_e_f[e_idx, f_idx]

        for sentence_idx, e_idx in enumerate(e_sentence):
            for f_idx in f_sentence:
                delta = t_e_f[e_idx, f_idx] / s_total_e[sentence_idx]
                counter[e_idx, f_idx] += delta
                total_f[f_idx] += delta

    return total_f, counter


def find_match(words_dict, t_e_f, french_word):
    f_idx = words_dict['french'][french_word]
    e_idx = np.argmax(t_e_f[:, f_idx])
    return list(words_dict['english'].keys())[e_idx]


def evaluate_results(numeric_corpus, t_e_f, output_path, low_prob=0):
    output_file = open(output_path, 'w')
    for e_sentence, f_sentence in tqdm(zip(numeric_corpus['english'], numeric_corpus['french'])):
        cur_str = ''
        for e_pos, e_idx in enumerate(e_sentence):
            max_prob = 0
            best_match = ()
            for f_pos, f_idx in enumerate(f_sentence):
                cur_prob = t_e_f[e_idx, f_idx]
                if cur_prob > max_prob and cur_prob > low_prob:
                    max_prob = cur_prob
                    best_match = (e_pos, f_pos)
            if max_prob != 0:
                cur_str += f'{best_match[1]}-{best_match[0]} '

        output_file.write(cur_str + '\n')
    output_file.close()


if __name__ == "__main__":
    english_corpus_path = r'./word-alignment/data/hansards.e'
    french_corpus_path = r'./word-alignment/data/hansards.f'
    output_test = r'results_model1/results_model_1.txt'
    t_e_f_path = r'results_model1/tef_15_iter_model_1.npy'

    n_iter = 5

    corpus, _ = read_corpus(english_corpus_path, french_corpus_path)
    # a set with all of the words (they will be referred by index)
    # clean_corpus(corpus)
    words_dict, n_words = format_data(corpus)

    # implement the arrays as 2d numpy array
    total_french_words = len(list(itertools.chain.from_iterable(corpus['french'])))
    t_e_f = np.zeros((n_words['english'], n_words['french'])) + 1 / float(total_french_words)

    # convert to numbers
    numeric_corpus = convert_corpus_to_numeric(corpus, words_dict)
    t_e_f_prev = 0

    has_changed = False
    # calculate translation
    if os.path.exists(t_e_f_path):
        with open(t_e_f_path, 'rb') as f:
            t_e_f = np.load(f)
    else:
        for i in range(n_iter):
            counter = np.zeros((n_words['english'], n_words['french']))
            total_f = np.zeros(n_words['french'])
            total_f, counter = evaluate_sentence(numeric_corpus, total_f, counter)
            t_e_f = counter / total_f
            print(f'\ndiff is: {(np.abs(t_e_f[:1000, :1000] - t_e_f_prev)).sum()}')
            t_e_f_prev = t_e_f[:1000, :1000]
        print(f'saving output file to {t_e_f_path}')
        with open(t_e_f_path, 'wb') as f:
            np.save(f, t_e_f)

    evaluate_results(numeric_corpus, t_e_f, output_test)
    print('done!')
