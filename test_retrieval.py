from backend.retriever import load_vector_store, retrieve
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
vector_store = load_vector_store(str(BASE_DIR / "vector_db" / "index.pkl"))

query = "What is Assadullah's email?"

results = retrieve(vector_store, query)

for r in results:
    print(r)
