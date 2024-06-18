#scorer.py
import numpy as np

def avg_wer(wer_scores, combined_ref_len):
    return float(sum(wer_scores)) / float(combined_ref_len)

def _levenshtein_distance(ref, hyp):
    m, n = len(ref), len(hyp)

    # Special cases
    if ref == hyp:
        return 0
    if m == 0:
        return n
    if n == 0:
        return m

    if m < n:
        ref, hyp = hyp, ref
        m, n = n, m

    # Use O(min(m, n)) space
    distance = np.zeros((2, n + 1), dtype=np.int32)

    # Initialize distance matrix
    for j in range(n + 1):
        distance[0][j] = j

    # Calculate Levenshtein distance
    for i in range(1, m + 1):
        prev_row_idx = (i - 1) % 2
        cur_row_idx = i % 2
        distance[cur_row_idx][0] = i
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                distance[cur_row_idx][j] = distance[prev_row_idx][j - 1]
            else:
                s_num = distance[prev_row_idx][j - 1] + 1
                i_num = distance[cur_row_idx][j - 1] + 1
                d_num = distance[prev_row_idx][j] + 1
                distance[cur_row_idx][j] = min(s_num, i_num, d_num)

    return distance[m % 2][n]

def word_errors(reference, hypothesis, ignore_case=False, delimiter=' '):
    if ignore_case:
        reference = reference.lower()
        hypothesis = hypothesis.lower()

    ref_words = reference.split(delimiter)
    hyp_words = hypothesis.split(delimiter)

    edit_distance = _levenshtein_distance(ref_words, hyp_words)
    return float(edit_distance), len(ref_words)

def char_errors(reference, hypothesis, ignore_case=False, remove_space=False):
    if ignore_case:
        reference = reference.lower()
        hypothesis = hypothesis.lower()

    join_char = ' ' if not remove_space else ''
    reference = join_char.join(filter(None, reference.split(' ')))
    hypothesis = join_char.join(filter(None, hypothesis.split(' ')))

    edit_distance = _levenshtein_distance(reference, hypothesis)
    return float(edit_distance), len(reference)

def wer(reference, hypothesis, ignore_case=False, delimiter=' '):
    edit_distance, ref_len = word_errors(reference, hypothesis, ignore_case, delimiter)
    if ref_len == 0:
        raise ValueError("Reference's word number should be greater than 0.")
    return float(edit_distance) / ref_len

def cer(reference, hypothesis, ignore_case=False, remove_space=False):
    edit_distance, ref_len = char_errors(reference, hypothesis, ignore_case, remove_space)
    if ref_len == 0:
        raise ValueError("Length of reference should be greater than 0.")
    return float(edit_distance) / ref_len