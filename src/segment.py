import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import torch
from transformers import AutoModelForTokenClassification
import torch.nn.functional as F

from utils import *


SEG_MODEL = 'dmis-lab/biosyn-biobert-bc5cdr-disease'


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val) for key, val in self.encodings[idx].items()}
        return item

    def __len__(self):
        return len(self.encodings)


def predict(inp, model, device, batch_size=8):
    test_loader = DataLoader(TestDataset(inp),
                             batch_size=batch_size, shuffle=False)
    predictions = []
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        p = F.sigmoid(model(**batch).logits).detach().cpu().numpy()
        predictions.append(p)

    return np.concatenate(predictions, axis=0)


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


def segment(notes):
    model = AutoModelForTokenClassification.from_pretrained('assets/biobert', num_labels=1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _ = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained('assets/biobert-tokenizer')

    thresh = 0.20603015075376885

    predictions = []
    for note in notes.itertuples():
        note_predictions = {'note_id': [], 'start': [], 'end': [], 'concept': [], 'scores': []}

        note_id = note.note_id
        raw_text = note.text

        inp = create_data(preprocess_text(raw_text), tokenizer)
        pred_probs = predict(inp, model, device, batch_size=8)
        pred_probs = np.squeeze(pred_probs, -1)
        pred = (pred_probs > thresh).astype('uint8')

        named_ents = set()
        seen_spans = set()
        for p, x, q in zip(pred, inp, pred_probs):

            # get entity strings
            tokens = np.array(x['input_ids'], 'uint32')
            token_spans = get_sequential_spans(p)
            entity_tokens = [tokens[s] for s in token_spans]
            entity_scores = [q[s].mean() for s in token_spans]

            # convert to strings
            pred_ent = [tokenizer.decode(t) for t in entity_tokens]

            # search for all occurences of each ent
            for ent, score in zip(pred_ent, entity_scores):
                # filter predicted ent strings (e.g. single char, article, ..)
                if (ent not in named_ents) and (len(ent) > 2):

                    for s, e in find_all_substrings(ent.lower(), raw_text.lower()):

                        if not is_overlap(seen_spans, (s, e)):

                            note_predictions['note_id'].append(note_id)
                            note_predictions['start'].append(s)
                            note_predictions['end'].append(e)
                            note_predictions['concept'].append(ent)
                            note_predictions['scores'].append(score)
                            named_ents.add(ent)
                            seen_spans.add((s, e))

        note_predictions = pd.DataFrame(note_predictions)
        note_predictions = note_predictions.sort_values('scores', ascending=False).iloc[:1000]

        predictions.append(note_predictions)

    predictions = pd.concat(predictions).reset_index(drop=True)
    return predictions
