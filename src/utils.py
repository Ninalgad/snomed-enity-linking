import numpy as np


def create_data(text, tokenizer, entities=None, max_blocks=8):
    has_labels = entities is not None
    inp, tar = [], []
    embeddings = tokenizer(text, padding="max_length", truncation=True,
                           max_length=max_blocks * 512)

    for i in range(max_blocks):
        x = {k: v[i * 512: (i + 1) * 512] for (k, v) in embeddings.items()}

        if x['input_ids'][0] == tokenizer.pad_token_id:
            break

        if has_labels:
            y = get_seg_label(entities, x['input_ids'], tokenizer)
        else:
            y = np.ones(1, 'uint8')

        if y.sum() != 0:
            inp.append(x)
            tar.append(y)

    if has_labels:
        return inp, tar
    return inp


def tokenize_entities(entities, tokenizer):
    results = []
    for e in entities:
        ent_tok = tokenizer.encode(e)[1:-1]
        results.append(ent_tok)

    return results


def get_seg_label(entities, tokens, tokenizer):
    n_tokens = len(tokens)
    label = np.zeros(n_tokens, 'uint8')
    ent_tokens = tokenize_entities(entities, tokenizer)

    for e in ent_tokens:
        n = len(e)

        for i in range(n, n_tokens - n):
            if tokens[i: i + n] == e:
                label[i: i + n] = 1

    return label


def preprocess_text(t):
    i = t.index('Sex:')
    t = t[i:]
    # t = t[400:]
    # t = t.split(' ')[1:]
    # t = " ".join(t)

    t = t.replace('\n \nAttending: ___.\n \n', '\n\n')
    t = t.replace('___', '_')
    t = t.replace('\n \n', ' \n ')
    t = t.replace(' \n', ' \n ')
    t = t.replace('  ', ' ')

    t = t.lower()
    return t


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


def find_all_substrings(sub, string):
    # https://stackoverflow.com/a/3874760
    """
    >>> text = "Allowed Hello Hollow"
    >>> tuple(findall('ll', text))
    (1, 10, 16)
    """
    index = 0 - len(sub)
    try:
        while True:
            index = string.index(sub, index + len(sub))
            yield index, index + len(sub)
    except ValueError:
        pass
