from pathlib import Path

import pickle
import pandas as pd
from loguru import logger
import torch

from transformers import AutoModelForTokenClassification, transformers, AutoTokenizer

from segment import segment

NOTES_PATH = Path("data/test_notes.csv")
SUBMISSION_PATH = Path("submission.csv")
LINKER_PATH = Path("assets/linker.pickle")
CER_MODEL_PATH = Path("assets/cer_model")

CONTEXT_WINDOW_WIDTH = 12
MAX_SEQ_LEN = 256


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


def main():
    # columns are note_id, text
    logger.info("Reading in notes data.")
    notes = pd.read_csv(NOTES_PATH)
    logger.info(f"Found {notes.shape[0]} notes.")

    # Load model components
    logger.info("Loading CER pipeline.")
    cer_model = AutoModelForTokenClassification.from_pretrained(CER_MODEL_PATH, num_labels=1)
    cer_tokenizer = AutoTokenizer.from_pretrained(
        CER_MODEL_PATH  # , model_max_length=max_seq_len
    )

    logger.info("Loading Linker")
    with open(LINKER_PATH, "rb") as f:
        linker = pickle.load(f)

    # Process one note at a time...
    logger.info("Processing notes.")
    thresh = 0.3877551020408163
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df_pred = segment(notes, thresh, cer_model, cer_tokenizer, device, batch_size=8)

    predictions = []
    for note_id, note_pred_df in df_pred.groupby('note_id'):
        note_pred_df = note_pred_df[(note_pred_df['end'] - note_pred_df['start']) > 1]
        concept_id = []
        seen_spans = set()
        for row in note_pred_df.itertuples():
            span = (row.start, row.end)
            if not is_overlap(seen_spans, span):
                concept_id.append(linker.link(row))
        note_pred_df['concept_id'] = concept_id
        predictions.append(note_pred_df)

    logger.info(f"Generated {len(predictions)} annotated spans.")
    predictions = pd.concat(predictions)
    predictions.to_csv(SUBMISSION_PATH, index=False)
    logger.info("Finished.")


if __name__ == "__main__":
    main()
