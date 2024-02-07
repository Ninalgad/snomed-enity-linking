from pathlib import Path

import pickle
import pandas as pd
from loguru import logger
import torch

from transformers import AutoModelForTokenClassification, AutoTokenizer

from segment import segment, segment_tokens
from linker import Linker


NOTES_PATH = Path("data/test_notes.csv")
SUBMISSION_PATH = Path("submission.csv")
LINKER_PATH = Path("assets/linker.pickle")

CONTEXT_WINDOW_WIDTH = 12
MAX_SEQ_LEN = 256

THRESH = [0.55102041, 0.2244898, 0.28571429, 0.16326531, 0.10204082]
SEEDS = [1, 423423, 346436, 9679677, 1233]


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


def load_algorithm(model_path):
    model = AutoModelForTokenClassification.from_pretrained(model_path, num_labels=1)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path  # , model_max_length=max_seq_len
    )
    return model, tokenizer


def main():
    # columns are note_id, text
    logger.info("Reading in notes data.")
    notes = pd.read_csv(NOTES_PATH)
    logger.info(f"Found {notes.shape[0]} notes.")

    logger.info("Loading Linker")
    with open(LINKER_PATH, "rb") as f:
        linker = pickle.load(f)

    # Process one note at a time...
    logger.info("Segmenting notes.")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    prediction_agg_map = dict()
    for seed, thresh in zip(SEEDS, THRESH):
        # Load model components
        cer_model, cer_tokenizer = load_algorithm(f"assets/cer-{seed}")
        cer_model.to(device)
        predictions_map = segment_tokens(notes, thresh, cer_model, cer_tokenizer, device)

        # Aggregate predictions
        for note_id, pred in predictions_map.items():
            prediction_agg_map[note_id] = prediction_agg_map.get(note_id, 0.) + pred

    # Majority voting ensemble
    thresh = int(len(THRESH) // 2)
    for note_id, pred in prediction_agg_map.items():
        prediction_agg_map[note_id] = (pred >= thresh).astype('uint8')

    # Create segmentation spans
    logger.info("Generating spans.")
    df_spans = segment(notes, prediction_agg_map, cer_tokenizer)

    # Label spans
    logger.info("Labeling spans.")
    predictions = []
    for note_id, note_pred_df in df_spans.groupby('note_id'):
        note_pred_df = note_pred_df[(note_pred_df['end'] - note_pred_df['start']) > 1]
        seen_spans = set()
        for row in note_pred_df.itertuples():
            span = (row.start, row.end)
            if not is_overlap(seen_spans, span):
                seen_spans.add(span)
                pred = {
                  'note_id': note_id,
                  'start': row.start,
                  'end': row.end,
                  'concept_id': linker.link(row),
                }
                predictions.append(pred)

    logger.info(f"Generated {len(predictions)} annotated spans.")
    predictions = pd.DataFrame(predictions)
    predictions.to_csv(SUBMISSION_PATH, index=False)
    logger.info("Finished.")


if __name__ == "__main__":
    main()
