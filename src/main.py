from pathlib import Path
from loguru import logger
import pandas as pd
import numpy as np

from segment import segment
from aer import disambiguation

NOTES_PATH = Path("data/test_notes.csv")
SUBMISSION_PATH = Path("submission.csv")

if __name__ == "__main__":
    # columns are note_id, text
    logger.info("Reading in notes data.")
    notes = pd.read_csv(NOTES_PATH)
    logger.info(f"Found {notes.shape[0]} notes.")

    predictions = segment(notes)

    concept_ids = np.ones(len(predictions), 'uint32') * 281900007  # default with most common concept_id
    entities = list(predictions['concept'].unique())
    cid_predictions = disambiguation(entities)

    for cid, e in zip(cid_predictions, entities):
        concept_ids[predictions['concept'] == e] = cid

    predictions['concept_id'] = concept_ids
    predictions = predictions.drop('concept', axis=1)

    logger.info(f"Generated {len(predictions)} annotated spans.")
    predictions.to_csv(SUBMISSION_PATH, index=False)
