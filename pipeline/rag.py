from .chunking import extract_chunks, split_text_to_paragraphs
from .extraction import extract_text_from_pdf
from .embedding import embed_query
from .generation import build_prompt, stream_response
from .retrieval import build_faiss_index, hybrid_search_faiss


async def rag(pdf: str, query: str, top_k: int = 1, use_cross_enc: bool = True):
    text = extract_text_from_pdf(pdf)
    paragraphs = split_text_to_paragraphs(text)
    chunks = extract_chunks(paragraphs)

    index, _ = build_faiss_index(chunks)
    query_embedding = embed_query(query)

    retrieved_chunks = hybrid_search_faiss(
        query=query,
        query_embedding=query_embedding,
        chunks=chunks,
        index=index,
        use_cross_enc=use_cross_enc,
        top_k=top_k,
    )

    messages = build_prompt(query=query, chunks=retrieved_chunks)
    async for token in stream_response(messages):
        yield token


rag_model = rag
