from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

from transformers.utils import logging as transformers_logging
from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_MODEL_ERROR_MESSAGE = (
    "The embedding model could not be loaded. Connect to the internet once and run "
    "`python -m src.ingest` to download the model, then restart the application."
)

transformers_logging.set_verbosity_error()



@lru_cache(maxsize=1)
def load_embedding_model() -> SentenceTransformer:
    """Load the embedding model once and reuse it."""
    try:
        return SentenceTransformer(EMBEDDING_MODEL_NAME, local_files_only=True)
    except Exception:
        try:
            return SentenceTransformer(EMBEDDING_MODEL_NAME)
        except Exception as error:
            raise RuntimeError(EMBEDDING_MODEL_ERROR_MESSAGE) from error


def embed_texts(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Embed a list of texts in batches."""
    if not texts:
        return np.empty((0, 384), dtype="float32")

    model = load_embedding_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return embeddings.astype("float32")
