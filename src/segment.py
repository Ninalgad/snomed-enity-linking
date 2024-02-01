import pandas as pd
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForTokenClassification

from utils import *


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val) for key, val in self.encodings[idx].items()}
        return item

    def __len__(self):
        return len(self.encodings)


def predict_segmentation(inp, model, device, batch_size=8):
    test_loader = DataLoader(TestDataset(inp),
                             batch_size=batch_size, shuffle=False)
    predictions = []
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        p = F.sigmoid(model(**batch).logits).detach().cpu().numpy()
        predictions.append(p)

    return np.concatenate(predictions, axis=0)


def create_data(text, tokenizer, seq_len=512):
    tokens = tokenizer(text, add_special_tokens=False)
    _token_batches = {k: [padd_seq(x, seq_len) for x in batch_list(v, seq_len)]
                      for (k, v) in tokens.items()}
    n_batches = len(_token_batches['input_ids'])
    return [{k: v[i] for k, v in _token_batches.items()}
            for i in range(n_batches)]


def segment_tokens(notes, model, tokenizer, device, batch_size=8):
    predictions = {}
    for note in notes.itertuples():
        raw_text = note.text.lower()

        inp = create_data(raw_text, tokenizer)
        pred_probs = predict_segmentation(inp, model, device, batch_size=batch_size)
        pred_probs = np.squeeze(pred_probs, -1)
        pred_probs = np.concatenate(pred_probs)

        predictions[note] = pred_probs

    return predictions


def segment(notes, thresh, model, tokenizer, device, predictions_prob_map=None, batch_size=8):
    predictions = []

    if predictions_prob_map is None:
        predictions_prob_map = segment_tokens(notes, model, tokenizer, device, batch_size)

    period_token = tokenizer.encode(".", add_special_tokens=False).pop()

    for note in notes.itertuples():
        note_predictions = {'note_id': [], 'start': [], 'end': [], 'entity': []}

        note_id = note.note_id
        raw_text = note.text.lower()

        raw_tokens = tokenizer.encode(raw_text, add_special_tokens=False)
        raw_tokens = np.array(raw_tokens, 'uint32')

        pred_probs = predictions_prob_map[note][:len(raw_tokens)]
        pred = (pred_probs > thresh).astype('uint8')

        sentences = split_str_with_delim(raw_text, ".")
        tok_sentence = split_arr_with_delim(raw_tokens, period_token)
        pred_sentence = []
        pointer = 0
        for s in tok_sentence:
            n = len(s)
            pred_sentence.append(pred[pointer:pointer + n])
            pointer += n

        pointer = 0
        for (sen, pred_sen, tok_sen) in zip(sentences, pred_sentence, tok_sentence):
            if pred_sen.max() == 0:
                pointer += len(sen)
                continue

            tok_sen = np.array(tok_sen, 'uint32')
            # get predicted entities
            spans = get_sequential_spans(pred_sen)
            tok_ent = [tok_sen[s] for s in spans]
            pred_ent = [tokenizer.decode(t) for t in tok_ent]

            seen_spans = set()
            for ent in pred_ent:
                if ent not in sen:
                    # if no exact reverse token mapping:
                    # get best approx using substrings of same word length
                    ent = best_matching_substring(sen, ent)

                if ent not in sen:
                    s = 0
                    e = len(ent) - 1
                else:
                    # get first
                    s = sen.index(ent)
                    e = s + len(ent)

                if (e - s) <= 1:
                    s = 0
                    e = len(ent) - 1

                span = (s + pointer, e + pointer)

                # assert sen[s:e] == raw_text[span[0]: span[1]], (ent, sen[s:e], raw_text[span[0]: span[1]])

                if not is_overlap(seen_spans, span):
                    note_predictions['note_id'].append(note_id)
                    note_predictions['start'].append(span[0])
                    note_predictions['end'].append(span[1])
                    note_predictions['entity'].append(sen[s:e])
                    seen_spans.add(span)

            pointer += len(sen)
            # print(pointer, len(raw_text))

        note_predictions = pd.DataFrame(note_predictions)
        predictions.append(note_predictions)

    predictions = pd.concat(predictions).reset_index(drop=True)
    return predictions
