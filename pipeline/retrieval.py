import numpy as np
import faiss
from rank_bm25 import BM25Okapi

from .embedding import tokenizer, embed_documents
from .reranking import rerank_documents


def build_faiss_index(chunks: list[str]) -> tuple:
    embeddings = embed_documents(chunks)
    embeddings_np = np.array(embeddings).astype("float32")
    dimension = embeddings_np.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings_np)
    return index, embeddings_np


def hybrid_search_faiss(
    query: str,
    query_embedding: np.ndarray,
    chunks: list[str],
    index,
    use_cross_enc: bool = True,
    top_k: int = 5,
) -> list[str]:
    query_np = np.array([query_embedding]).astype("float32")
    distances, indices = index.search(query_np, k=10)
    dense_ids = [f"chunk_{i}" for i in indices[0]]

    tokenized_chunks = [tokenizer.tokenize(chunk) for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    tokenized_query = tokenizer.tokenize(query)
    bm25_scores = bm25.get_scores(tokenized_query)
    top_sparse_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:20]
    sparse_ids = [f"chunk_{i}" for i in top_sparse_indices]

    def reciprocal_rank_fusion(dense_ids: list[str], sparse_ids: list[str], k: int = 60) -> list[str]:
        scores = {}
        for rank, doc_id in enumerate(dense_ids):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
        for rank, doc_id in enumerate(sparse_ids):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
        return sorted(scores, key=scores.get, reverse=True)

    def ids_to_texts(ids: list[str]) -> list[str]:
        return [
            chunks[int(doc_id.replace("chunk_", ""))]
            for doc_id in ids
            if int(doc_id.replace("chunk_", "")) < len(chunks)
        ]

    final_ids = reciprocal_rank_fusion(dense_ids, sparse_ids)
    candidate_texts = ids_to_texts(final_ids[:10])

    if use_cross_enc:
        return rerank_documents(query, candidate_texts, top_k=top_k)

    return ids_to_texts(final_ids[:top_k])
