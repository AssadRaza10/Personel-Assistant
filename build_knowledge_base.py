import argparse
import os
import pickle

from backend.knowledge_loader import load_documents
from backend.text_splitter import split_text
from backend.embedding_generator import generate_embeddings
from backend.vector_store import VectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS vector store from text files.")
    parser.add_argument(
        "--source-dir",
        default="knowledge_base",
        help="Directory containing .txt knowledge files",
    )
    parser.add_argument(
        "--output",
        default="vector_db/index.pkl",
        help="Path to write the pickled vector store",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=300,
        help="Approximate number of words per chunk",
    )

    args = parser.parse_args()

    if not os.path.isdir(args.source_dir):
        raise FileNotFoundError(
            f"Knowledge base directory not found: {args.source_dir}"
        )

    documents = load_documents(args.source_dir)
    if not documents:
        raise RuntimeError(
            f"No .txt documents found in {args.source_dir}. Add files and run again."
        )

    chunks = []
    for doc in documents:
        chunks.extend(split_text(doc, chunk_size=args.chunk_size))

    embeddings = generate_embeddings(chunks)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    dimension = embeddings.shape[1]
    vector_store = VectorStore(dimension)
    vector_store.add(embeddings, chunks)

    with open(args.output, "wb") as f:
        pickle.dump(vector_store, f)

    print(f"Knowledge base created successfully at {args.output}!")


if __name__ == "__main__":
    main()
