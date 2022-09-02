import os
import time
from functools import wraps

import numpy as np
from tqdm import tqdm


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
def train_model_2(t_e_f, q, n_iter, numeric_corpus, max_len, n_words):
    t_e_f_prev = 0
    np_type = np.float32
    for i in range(n_iter):
        # initialize counters
        counter = np.zeros((n_words['english'], n_words['french']), dtype=np_type)
        total_f = np.zeros((n_words['french']), dtype=np_type)

        delta = np.zeros((max_len['english'], max_len['french']), dtype=np_type)
        counter_4d = np.zeros((max_len['french'], max_len['english'], max_len['french'], max_len['english']),
                              dtype=np_type)
        counter_3d = np.zeros((max_len['english'], max_len['french'], max_len['english']), dtype=np_type)

        # run over all sentences
        for e_sentence, f_sentence in tqdm(zip(numeric_corpus['english'], numeric_corpus['french'])):
            m = len(e_sentence) - 1
            l = len(f_sentence) - 1
            s_total_e = np.zeros(len(e_sentence))
            q_t_f = []
            for e_pos, e_idx in enumerate(e_sentence):
                for f_pos, f_idx in enumerate(f_sentence):
                    cur_qtf = q[f_pos, e_pos, l, m] * t_e_f[e_idx, f_idx]
                    q_t_f.append(cur_qtf)
                    s_total_e[e_pos] += cur_qtf

            q_t_f_idx = 0
            for e_pos, e_idx in enumerate(e_sentence):
                for f_pos, f_idx in enumerate(f_sentence):
                    delta[e_pos, f_pos] = q_t_f[q_t_f_idx] / s_total_e[e_pos]
                    counter[e_idx, f_idx] += delta[e_pos, f_pos]
                    total_f[f_idx] += delta[e_pos, f_pos]
                    counter_3d[e_pos, l, m] += delta[e_pos, f_pos]
                    counter_4d[f_pos, e_pos, l, m] += delta[e_pos, f_pos]
                    q_t_f_idx += 1

        # calculate current q and t_e_f
        t_e_f = counter / total_f
        q = counter_4d / counter_3d
        print(f'\ndiff is: {(np.abs(t_e_f[:1000, :1000] - t_e_f_prev)).sum()}')
        t_e_f_prev = t_e_f[:1000, :1000]

    return t_e_f, q