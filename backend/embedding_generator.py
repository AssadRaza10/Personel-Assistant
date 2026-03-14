"""Embedding generator helper.

This module exports a shared SentenceTransformer model instance and a helper function.
The import-time model load may raise ImportError if sentence-transformers is not installed.
"""

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    raise ImportError(
        "Missing dependency 'sentence-transformers'. Install it with `pip install sentence-transformers`."
    ) from e

model = SentenceTransformer("all-MiniLM-L6-v2")


def generate_embeddings(text_chunks):
    """Generate embeddings for a list of text chunks."""
    return model.encode(text_chunks)
