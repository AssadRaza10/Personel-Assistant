from backend.retriever import load_vector_store, retrieve

vector_store = load_vector_store("vector_db/index.pkl")

query = "What is Assadullah's email?"

results = retrieve(vector_store, query)

for r in results:
    print(r)
