from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
import pickle
import torch
import numpy as np

from trie import Trie


def disambiguation(entities, batch_size=8):
    model = AutoModelForSeq2SeqLM.from_pretrained('assets/t5')
    tokenizer = AutoTokenizer.from_pretrained('assets/t5-tokenizer')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _ = model.to(device)

    if tokenizer.bos_token_id is None:
        tokenizer.bos_token_id = tokenizer.pad_token_type_id

    with open('assets/fsn2conceptid-t5-base.pkl', 'rb') as f:
        fsn2conceptid = pickle.load(f)

    fsns = np.load('assets/fsn.npy', allow_pickle=True)

    trie = Trie([[tokenizer.bos_token_id] + tokenizer.encode(x.lower())
                 for x in fsns])

    predicted_concept_ids = []
    i = 0
    while i < len(entities):
        ent_batch = entities[i:i + batch_size]
        batch = tokenizer(ent_batch, padding="max_length", truncation=True,
                          return_tensors='pt', max_length=32)
        batch = {k: v.to(device) for (k, v) in batch.items()}
        with torch.no_grad():
            output = model.generate(
                **batch,
                prefix_allowed_tokens_fn=lambda batch_id, sent: trie.get(sent.tolist()),
                max_new_tokens=100,
                # num_beams=10, do_sample=True, remove_invalid_values=True
            )
        generated_fsn = tokenizer.batch_decode(output, skip_special_tokens=True)
        predicted_concept_ids += [fsn2conceptid[x.lower()] for x in generated_fsn]

        i += batch_size

    return predicted_concept_ids
