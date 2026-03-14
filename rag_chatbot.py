from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import pickle
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# -----------------------------
# Load LLM Model (FLAN-T5)
# -----------------------------

print("Loading FLAN-T5 model...")

model_path = BASE_DIR / "models" / "flan_t5"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)


# -----------------------------
# Load Embedding Model
# -----------------------------

print("Loading embedding model...")

embed_model = SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------
# Load Vector Database
# -----------------------------

print("Loading knowledge base...")

vector_path = BASE_DIR / "vector_db" / "index.pkl"
with open(vector_path, "rb") as f:
    vector_store = pickle.load(f)


# -----------------------------
# Retrieve Context from Vector DB
# -----------------------------

def retrieve_context(query, k=3):

    query_embedding = embed_model.encode([query])

    results = vector_store.search(query_embedding, k)

    context = "\n".join(results[:2])

    return context


# -----------------------------
# Prompt Builder
# -----------------------------

def build_prompt(context, question):

    prompt = f"""
Answer the question using the context below.

Context:
{context}

Question:
{question}

Answer:
"""

    return prompt


# -----------------------------
# Generate Answer
# -----------------------------

def generate_answer(prompt):

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=80
    )

    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return answer


# -----------------------------
# Chat Loop
# -----------------------------

print("\nAssistant Ready! Type 'exit' to quit.\n")

while True:

    question = input("You: ")

    if question.lower() in ["exit", "quit"]:
        break

    context = retrieve_context(question)

    prompt = build_prompt(context, question)

    answer = generate_answer(prompt)

    print("\nAssistant:", answer)
    print("\n")