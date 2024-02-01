from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import numpy as np
import torch


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.x = encodings

    def __getitem__(self, idx):
        x = {key: torch.tensor(val) for key, val in self.x[idx].items()}
        return x

    def __len__(self):
        return len(self.x)


def create_rerank_data(entities, fsn, tokenizer):
    dataset = []
    for (e, f) in zip(entities, fsn):
        prompt = f"{e} is an example of {f}"
        inp = tokenizer(prompt.lower(), padding="max_length", truncation=True,
                        max_length=64)
        dataset.append(inp)
    return dataset


def predict(loader, model, device):
    pred = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        p = torch.nn.Sigmoid()(model(**batch).logits).detach().cpu().numpy()
        p = np.squeeze(p, -1)
        p = p.astype('float32')
        pred.append(p)

    return np.concatenate(pred)


def predict_rerank(entities, candidate_fsn, batch_size=32):
    model = AutoModelForSequenceClassification.from_pretrained('assets/biobert-rr', num_labels=1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _ = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained('assets/biobert-rr-tokenizer')

    predictions = []
    scores = []

    for ent, fsn in zip(entities, candidate_fsn):
        dataset = create_rerank_data([ent for _ in fsn], fsn, tokenizer)
        test_loader = DataLoader(TestDataset(dataset), batch_size=batch_size, shuffle=False)
        probs = predict(test_loader, model, device)
        i = np.argmax(probs)
        predictions.append(i)
        scores.append(probs[i])

    return predictions, scores
