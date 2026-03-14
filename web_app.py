import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import pickle
import numpy as np
from pathlib import Path
from pypdf import PdfReader

BASE_DIR = Path(__file__).resolve().parent


# -------------------------
# PDF TEXT EXTRACTION
# -------------------------

def extract_text_from_pdf(uploaded_file):

    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:
        try:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

        except Exception as e:
            print("Skipping page due to extraction error:", e)

    return text


# -------------------------
# TEXT CHUNKING
# -------------------------

def split_text(text, chunk_size=500):

    chunks = []

    words = text.split()

    for i in range(0, len(words), chunk_size):

        chunk = " ".join(words[i:i+chunk_size])

        chunks.append(chunk)

    return chunks


# -------------------------
# PAGE SETTINGS
# -------------------------

st.set_page_config(
    page_title="Personal AI Assistant",
    page_icon="🤖",
    layout="wide"
)


# -------------------------
# LOAD MODELS
# -------------------------

@st.cache_resource
def load_models():

    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    vector_path = BASE_DIR / "vector_db" / "index.pkl"
    with open(vector_path, "rb") as f:
        vector_store = pickle.load(f)

    return tokenizer, model, embed_model, vector_store


tokenizer, model, embed_model, vector_store = load_models()


# -------------------------
# RAG FUNCTIONS
# -------------------------

def retrieve_context(query, k=3):

    query_embedding = embed_model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    results = vector_store.search(query_embedding, k)

    context = "\n".join(results)

    return context


def build_prompt(context, question):

    prompt = f"""
    You are a helpful AI assistant.

    Use the context to answer the question clearly and concisely.

    If the context contains the answer, explain it in your own words.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    return prompt


def generate_answer(prompt):

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    outputs = model.generate(
        **inputs,
        max_new_tokens=80
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return answer


# -------------------------
# SESSION STATE
# -------------------------

if "conversations" not in st.session_state:
    st.session_state.conversations = {}

if "current_chat" not in st.session_state:
    st.session_state.current_chat = "Chat 1"
    st.session_state.conversations["Chat 1"] = []


# -------------------------
# SIDEBAR
# -------------------------

with st.sidebar:

    st.title("💬 Chats")

    if st.button("➕ New Chat"):

        chat_id = f"Chat {len(st.session_state.conversations)+1}"

        st.session_state.conversations[chat_id] = []

        st.session_state.current_chat = chat_id

    st.divider()

    for chat in st.session_state.conversations.keys():

        if st.button(chat):

            st.session_state.current_chat = chat


    st.divider()

    st.subheader("📄 Upload PDF")

    uploaded_pdf = st.file_uploader(
        "Upload a PDF file",
        type=["pdf"]
    )

    if uploaded_pdf:

        text = extract_text_from_pdf(uploaded_pdf)

        chunks = split_text(text)

        embeddings = embed_model.encode(chunks)
        embeddings = np.array(embeddings).astype("float32")

        vector_store.add(chunks, embeddings)

        st.success("PDF uploaded and indexed!")


# -------------------------
# MAIN CHAT AREA
# -------------------------

st.title("🤖 Personal AI Assistant")

chat_history = st.session_state.conversations[st.session_state.current_chat]


# Display previous messages
for msg in chat_history:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# -------------------------
# CHAT INPUT
# -------------------------

prompt = st.chat_input("Ask me something...")

if prompt:

    chat_history.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            context = retrieve_context(prompt)

            full_prompt = build_prompt(context, prompt)

            answer = generate_answer(full_prompt)

            st.markdown(answer)

    chat_history.append({"role": "assistant", "content": answer})