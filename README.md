# Personal AI Assistant

A simple Retrieval-Augmented Generation (RAG) assistant built with:

- **FLAN-T5** (via `transformers`) as the LLM
- **Sentence Transformers** for embedding
- **FAISS** as a vector store
- **Streamlit** for the web UI

---

## 🚀 Quick Start

### 1) Create & activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```powershell
pip install -r requirements.txt
```

### 3) Download the FLAN-T5 model (optional but recommended)

The repository can also use a locally cached model in `./models/flan_t5`.

```powershell
python download_model.py
```

### 4) Build the vector store from the knowledge base

Place `.txt` documents in `./knowledge_base/` and run:

```powershell
python build_knowledge_base.py
```

This creates `./vector_db/index.pkl`, which is required by the UI and the CLI chatbot.

---

## ▶️ Run the Streamlit web app

```powershell
streamlit run web_app.py
```

Then open the URL shown in your terminal (usually `http://localhost:8501`).

---

## 🧠 Other Usage

### Run the CLI chatbot

```powershell
python rag_chatbot.py
```

### Test retrieval from the vector store

```powershell
python test_retrieval.py
```

---

## 📁 Important Paths

- `knowledge_base/` – add `.txt` files here for retrieval
- `vector_db/index.pkl` – generated vector store (FAISS + chunks)
- `models/flan_t5/` – local FLAN-T5 model cache

---

## 📝 Notes

- If you update `knowledge_base/`, re-run `python build_knowledge_base.py` to rebuild the index.
- The app uses `pypdf` to extract text from PDFs uploaded via the Streamlit UI.
