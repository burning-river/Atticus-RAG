from sentence_transformers import CrossEncoder

cross_encoder_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2', backend="onnx")

def rerank_documents(query: str, documents: list[str], top_k: int = 5) -> list[str]:
    if not documents:
        return []

    rerank_pairs = [[query, doc] for doc in documents]
    scores = cross_encoder_model.predict(rerank_pairs)
    ranked = sorted(zip(documents, scores), key=lambda item: item[1], reverse=True)
    return [document for document, _score in ranked[:top_k]]
