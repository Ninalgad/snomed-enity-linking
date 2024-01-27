from pathlib import Path
from loguru import logger
import pandas as pd
import numpy as np

from segment import segment
from retrieve import retrieve

NOTES_PATH = Path("data/test_notes.csv")
SUBMISSION_PATH = Path("submission.csv")

if __name__ == "__main__":
    # columns are note_id, text
    logger.info("Reading in notes data.")
    notes = pd.read_csv(NOTES_PATH)
    logger.info(f"Found {notes.shape[0]} notes.")

    df_sctid = pd.read_csv('assets/sg.csv')
    thresh = 0.42703648794608556

    predictions = segment(notes, thresh, batch_size=16)

    entities = list(predictions['concept'].unique())
    sctid_index, _ = retrieve(
        entities, df_sctid['fsn'].str.lower().values
    )
    cid_predictions = df_sctid['sctid'].iloc[sctid_index].values

    concept_ids = np.ones(len(predictions), 'uint32') * 281900007  # default with most common concept_id
    for cid, e in zip(cid_predictions, entities):
        concept_ids[predictions['entity'] == e] = cid

    predictions['concept_id'] = concept_ids
    predictions = predictions.drop('entity', axis=1)

    logger.info(f"Generated {len(predictions)} annotated spans.")
    predictions.to_csv(SUBMISSION_PATH, index=False)
