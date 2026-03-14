import os
import pickle

from backend.embedding_generator import model as embed_model


def load_vector_store(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Vector store not found at '{path}'. Run `python build_knowledge_base.py` first."
        )

    with open(path, "rb") as f:
        return pickle.load(f)


def retrieve(vector_store, query: str, k: int = 3):
    query_embedding = embed_model.encode([query])
    results = vector_store.search(query_embedding, k)

    return results
