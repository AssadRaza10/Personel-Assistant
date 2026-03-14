import argparse
import os
import pickle
from pathlib import Path

from backend.knowledge_loader import load_documents
from backend.text_splitter import split_text
from backend.embedding_generator import generate_embeddings
from backend.vector_store import VectorStore

BASE_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS vector store from text files.")
    parser.add_argument(
        "--source-dir",
        default=str(BASE_DIR / "knowledge_base"),
        help="Directory containing .txt knowledge files",
    )
    parser.add_argument(
        "--output",
        default=str(BASE_DIR / "vector_db" / "index.pkl"),
        help="Path to write the pickled vector store",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=300,
        help="Approximate number of words per chunk",
    )

    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    output_path = Path(args.output)

    if not source_dir.is_dir():
        raise FileNotFoundError(
            f"Knowledge base directory not found: {source_dir}"
        )

    documents = load_documents(source_dir)
    if not documents:
        raise RuntimeError(
            f"No .txt documents found in {source_dir}. Add files and run again."
        )

    chunks = []
    for doc in documents:
        chunks.extend(split_text(doc, chunk_size=args.chunk_size))

    embeddings = generate_embeddings(chunks)

    os.makedirs(output_path.parent, exist_ok=True)

    dimension = embeddings.shape[1]
    vector_store = VectorStore(dimension)
    vector_store.add(embeddings, chunks)

    with open(output_path, "wb") as f:
        pickle.dump(vector_store, f)

    print(f"Knowledge base created successfully at {output_path}!")


if __name__ == "__main__":
    main()
