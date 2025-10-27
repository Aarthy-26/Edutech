
# langchain_helper.py
import os, json
from dotenv import load_dotenv
load_dotenv()

import requests
from typing import Tuple, List

from langchain.vectorstores import FAISS
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings

# Config
GROQ_API_KEY = os.getenv("GROQ_API_KEY")               # from .env
GROQ_API_URL = os.getenv("GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions")
# Example models: "llama-3.1-8b-instant", "llama-3.1-70b-versatile" (use one your account has)
GROQ_MODEL = os.getenv("GROQ_MODEL_NAME", "llama-3.1-8b-instant")

VECTORDB_FILE_PATH = "faiss_index"
CSV_FILE = "ed-tech_faqs.csv"
CSV_SOURCE_COL = "prompt"   # change if your CSV uses a different column name

# Embedding model (small, fast)
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

# ========== Vector DB functions ==========
def create_vector_db(csv_path: str = CSV_FILE, source_column: str = CSV_SOURCE_COL):
    """
    Create FAISS vector DB from CSV and save locally.
    """
    loader = CSVLoader(file_path=csv_path, source_column=source_column)
    docs = loader.load()
    vectordb = FAISS.from_documents(documents=docs, embedding=embeddings)
    vectordb.save_local(VECTORDB_FILE_PATH)
    return vectordb

def load_vector_db():
    return FAISS.load_local(VECTORDB_FILE_PATH, embeddings)

def retrieve_top_k(question: str, k: int = 4):
    """
    Return top-k documents (LangChain Document objects) and a single combined context string.
    """
    vectordb = load_vector_db()
    retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": k})
    docs = retriever.get_relevant_documents(question)
    parts = []
    for i, d in enumerate(docs, start=1):
        meta = getattr(d, "metadata", {}) or {}
        # Prefer a 'response' metadata field if present, otherwise use the page_content
        snippet = meta.get("response") or d.page_content
        parts.append(f"Source {i}:\n{snippet}\n")
    context = "\n---\n".join(parts)
    return context, docs

# ========== Groq call (HTTP) ==========
def call_groq(prompt: str, model: str = GROQ_MODEL, timeout: int = 30) -> Tuple[str, dict]:
    """
    Send prompt to Groq via HTTP and return (text, full_json_response).
    The exact response JSON can vary by Groq account/SDK version, so we try to parse common shapes.
    If parsing fails, the function returns the raw JSON string as text and the JSON as dict.
    """
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY missing in .env")

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
    "model": model,
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ],
    "temperature": 0.1
}


    resp = requests.post(GROQ_API_URL, json=payload, headers=headers, timeout=timeout)
    try:
        resp.raise_for_status()
    except Exception as e:
        # include body for easier debugging
        raise RuntimeError(f"Groq request failed: {e}\nStatus: {resp.status_code}\nBody: {resp.text}")

    j = resp.json()

    # Try some known response shapes (best-effort)
    # 1) Many APIs return something like: {"result":[{"content":[{"type":"output_text","text":"..."}]}]}
    try:
        # Common shape: result -> [0] -> content -> [0]['text']
        if isinstance(j, dict) and "result" in j and isinstance(j["result"], list) and j["result"]:
            res0 = j["result"][0]
            if isinstance(res0, dict) and "content" in res0 and isinstance(res0["content"], list) and res0["content"]:
                c0 = res0["content"][0]
                # possible keys: 'text', 'output', 'message', 'value'
                for k in ("text", "output", "message", "value"):
                    if k in c0 and isinstance(c0[k], str):
                        return c0[k], j

        # 2) Another possible shape: {"output": "the text"} or {"text": "..."}
        for k in ("output", "text", "generated_text"):
            if k in j and isinstance(j[k], str):
                return j[k], j

        # 3) Some responses nest under 'outputs' list
        if "outputs" in j and isinstance(j["outputs"], list) and j["outputs"]:
            out0 = j["outputs"][0]
            if isinstance(out0, dict):
                for k in ("content", "text", "payload"):
                    if k in out0:
                        if isinstance(out0[k], str):
                            return out0[k], j
                        elif isinstance(out0[k], dict) and "text" in out0[k]:
                            return out0[k]["text"], j
    except Exception:
        pass

    # fallback: return pretty-printed json so we can debug
    return json.dumps(j, indent=2, ensure_ascii=False), j

# ========== Top-level helper ==========
def answer_with_groq(question: str, k: int = 4, model: str = GROQ_MODEL):
    """
    Retrieve context from FAISS, build a prompt, call Groq, and return (answer_text, docs, raw_json).
    """
    context, docs = retrieve_top_k(question, k=k)

    combined_prompt = f"""You are a helpful assistant. Answer the user's QUESTION using ONLY the provided CONTEXT. 
If the answer is not present in the CONTEXT, respond with exactly: "I don't know."

CONTEXT:
{context}

QUESTION:
{question}

Answer (concise):"""

    text, raw = call_groq(combined_prompt, model=model)
    return text, docs, raw
