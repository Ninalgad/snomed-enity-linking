import numpy as np
from sentence_transformers import SentenceTransformer


def retrieve(query_texts, key_texts, block_size=512):
    model = SentenceTransformer('assets/sim')

    q = model.encode(query_texts)
    best_q_scores = [0 for _ in range(len(query_texts))]
    best_k_index = [0 for _ in range(len(query_texts))]

    i = 0
    while i < len(key_texts):
        text_block = key_texts[i: i + block_size]
        k = model.encode(text_block)
        v = np.matmul(q, k.T)  # q-s x k-bs

        if v.max() > max(best_q_scores):
            v_max = v.max(1)
            v_argmax = np.argmax(v, -1)
            for j, (batch_score, batch_argmax) in enumerate(zip(v_max, v_argmax)):
                if best_q_scores[j] < batch_score:
                    best_q_scores[j] = batch_score
                    best_k_index[j] = i + batch_argmax

        i += block_size

    return best_k_index, best_q_scores
