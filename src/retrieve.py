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


def retrieve_topk(topk, query_texts, key_texts, block_size=512):
    assert topk < block_size
    model = SentenceTransformer('assets/sim')

    q = model.encode(query_texts, normalize_embeddings=True)
    top_query_scores = [[] for _ in range(len(query_texts))]
    top_key_index = [[] for _ in range(len(query_texts))]

    n = 0
    while n < len(key_texts):
        text_block = key_texts[n: n + block_size]
        k = model.encode(text_block, normalize_embeddings=True)
        v = np.matmul(q, k.T)  # q-s x k-bs

        for i in range(v.shape[0]):
            if not top_query_scores[i]:
                x = list(v[i][:topk])
                top_query_scores[i] = x
                top_key_index[i] = list(range(len(x)))

            elif v[i].max() > max(top_query_scores[i]):
                for j in range(v.shape[1]):
                    if v[i][j] > min(top_query_scores[i]):
                        worst_index = np.argmin(top_query_scores[i])
                        del top_query_scores[i][worst_index]
                        del top_key_index[i][worst_index]

                        top_query_scores[i].append(v[i][j])
                        top_key_index[i].append(j + n)

        n += block_size

    return top_key_index, top_query_scores
