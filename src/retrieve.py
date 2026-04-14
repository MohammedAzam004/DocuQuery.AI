from __future__ import annotations

from pathlib import Path

from src.embed import embed_texts
from src.vector_store import load_index, search_index


MIN_RELEVANCE_SCORE = 0.4


def retrieve_chunks(question: str, index_folder: str | Path, k: int = 5) -> list[dict]:
    """Embed the question, search FAISS, and return top chunks."""
    index, chunks = load_index(index_folder=index_folder)

    # A query is just one sentence, so we wrap it in a list and embed it once.
    query_embedding = embed_texts([question], batch_size=1)
    retrieved_chunks = search_index(index=index, query_embedding=query_embedding, chunks=chunks, k=k)

    if not retrieved_chunks:
        return []

    # If the best match is weak, we treat the question as outside the knowledge base.
    if retrieved_chunks[0]["score"] < MIN_RELEVANCE_SCORE:
        return []

    return retrieved_chunks
