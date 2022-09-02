from tqdm import tqdm


def evaluate_results(numeric_corpus, t_e_f, output_path, low_prob=0):
    output_file = open(output_path, 'w')
    for e_sentence, f_sentence in tqdm(zip(numeric_corpus['english'], numeric_corpus['french'])):
        for e_pos, e_idx in enumerate(e_sentence):
            max_prob = 0
            best_match = ()
            for f_pos, f_idx in enumerate(f_sentence):
                cur_prob = t_e_f[e_idx, f_idx]
                if cur_prob > max_prob and cur_prob > low_prob:
                    max_prob = cur_prob
                    best_match = (e_pos, f_pos)
            if max_prob != 0:
                output_file.write(f'{best_match[1]}-{best_match[0]} ')

        output_file.write('\n')
    output_file.close()