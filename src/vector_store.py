from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np


INDEX_FILE_NAME = "index.faiss"
CHUNKS_FILE_NAME = "chunks.json"
DOCUMENTS_FILE_NAME = "documents.json"


def create_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Create a FAISS index for cosine similarity search."""
    if embeddings.size == 0:
        raise ValueError("No embeddings were created. Check your documents and chunks.")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index


def save_index(index: faiss.Index, chunks: list[dict], documents: list[dict], index_folder: str | Path) -> None:
    """Save the FAISS index and metadata to disk."""
    index_path = Path(index_folder)
    index_path.mkdir(parents=True, exist_ok=True)

    # Save the binary FAISS index.
    faiss.write_index(index, str(index_path / INDEX_FILE_NAME))

    # Save chunk metadata so we can show text, file names, and page numbers later.
    with (index_path / CHUNKS_FILE_NAME).open("w", encoding="utf-8") as file:
        json.dump(chunks, file, indent=2, ensure_ascii=False)

    # Save a small document summary for the CLI and Streamlit UI.
    with (index_path / DOCUMENTS_FILE_NAME).open("w", encoding="utf-8") as file:
        json.dump(documents, file, indent=2, ensure_ascii=False)


def load_index(index_folder: str | Path) -> tuple[faiss.Index, list[dict]]:
    """Load the saved FAISS index and chunk metadata."""
    index_path = Path(index_folder)
    faiss_path = index_path / INDEX_FILE_NAME
    chunks_path = index_path / CHUNKS_FILE_NAME

    if not faiss_path.exists() or not chunks_path.exists():
        raise FileNotFoundError(
            "FAISS index files were not found. Run the indexing step first with `python -m src.ingest`."
        )

    index = faiss.read_index(str(faiss_path))

    with chunks_path.open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    return index, chunks


def load_document_summary(index_folder: str | Path) -> list[dict]:
    """Load saved document summary for the UI."""
    documents_path = Path(index_folder) / DOCUMENTS_FILE_NAME

    if not documents_path.exists():
        return []

    with documents_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def search_index(
    index: faiss.Index,
    query_embedding: np.ndarray,
    chunks: list[dict],
    k: int = 5,
) -> list[dict]:
    """Search the index and return top matching chunks with metadata."""
    # FAISS returns both similarity scores and row positions.
    scores, positions = index.search(query_embedding, k)
    results: list[dict] = []

    for score, position in zip(scores[0], positions[0]):
        if position == -1:
            continue

        # Use the FAISS row position to pull back the saved metadata.
        match = chunks[position].copy()
        match["score"] = float(score)
        results.append(match)

    return results
