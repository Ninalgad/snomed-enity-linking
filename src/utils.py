import numpy as np
from difflib import SequenceMatcher


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def best_matching_substring(context, target):
    target = target.strip()
    sub_tok = target.split(" ")
    n = len(sub_tok)
    tok = context.split(" ")

    best_score, best_sub = 0, " ".join(target[:n])
    for i in range(len(tok) - n + 1):
        proposal = " ".join(tok[i:i + n])

        s = similar(proposal, target)
        if s > best_score:
            best_score = s
            best_sub = proposal

    return best_sub


def split_str_with_delim(text, d):
    return [e + d for e in text.split(d) if e]


def split_arr_with_delim(a, d):
    seqences = []
    cur = []
    for x in a:
        cur.append(x)
        if x == d:
            seqences.append(cur)
            cur = []
    return seqences


def get_sequential_spans(a):
    spans = []
    curr = []
    prev = 0

    for i, x in enumerate(a):
        if prev and x:  # continue span
            curr.append(i)
        elif x:  # start new span
            curr = [i]
        elif curr:  # save and start over if not empty
            spans.append(curr)
            curr = []

        prev = x

    if curr:
        spans.append(curr)

    return spans


# https://stackoverflow.com/a/3874760
def find_all_substrings(sub, string):
    """
    >>> text = "Allowed Hello Hollow"
    >>> tuple(find_all_substrings('ll', text))
    (1, 10, 16)
    """
    index = 0 - len(sub)
    try:
        while True:
            index = string.index(sub, index + len(sub))
            yield index, index + len(sub)
    except ValueError:
        pass


def is_overlap(existing_spans, new_span):
    for span in existing_spans:
        # Check if either end of the new span is within an existing span
        if (span[0] <= new_span[0] <= span[1]) or \
                (span[0] <= new_span[1] <= span[1]):
            return True
        # Check if the new span entirely covers an existing span
        if new_span[0] <= span[0] and new_span[1] >= span[1]:
            return True
    return False


def batch_list(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def padd_seq(seq, max_len):
    n = len(seq)
    if n >= max_len:
        return seq
    else:
        return np.pad(seq, (0, max_len - n))