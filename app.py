import os
import subprocess
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Consulting RAG Knowledge System", layout="wide")

st.title("RAG-based Knowledge System for Consulting Research")
st.caption("Upload PDFs → Build FAISS index → Ask questions → Get structured answers with traceable sources.")

DATA_RAW = Path("data_raw")
DATA_PROCESSED = Path("data_processed")
INDEX_DIR = Path("index")

DATA_RAW.mkdir(exist_ok=True)
DATA_PROCESSED.mkdir(exist_ok=True)
INDEX_DIR.mkdir(exist_ok=True)

with st.sidebar:
    st.header("Pipeline")
    emb_model = st.text_input("Embedding model", value="BAAI/bge-small-en-v1.5")
    mode = st.selectbox("Retrieval mode", ["hybrid", "vector"], index=0)
    top_k = st.slider("Top-K", 3, 12, 8)

st.subheader("1) Upload PDFs")
uploaded = st.file_uploader("Drop PDF files here", type=["pdf"], accept_multiple_files=True)
if uploaded:
    for f in uploaded:
        (DATA_RAW / f.name).write_bytes(f.getbuffer())
    st.success(f"Saved {len(uploaded)} file(s) into {DATA_RAW}/")

col1, col2 = st.columns(2)
with col1:
    if st.button("2) Ingest (clean + chunk)"):
        cmd = ["python", "src/ingest.py", "--input", str(DATA_RAW), "--output", str(DATA_PROCESSED / "chunks.jsonl")]
        res = subprocess.run(cmd, capture_output=True, text=True)
        st.code(res.stdout or res.stderr)

with col2:
    if st.button("3) Build Index (embed + FAISS)"):
        cmd = ["python", "src/embed_index.py", "--chunks", str(DATA_PROCESSED / "chunks.jsonl"), "--out", str(INDEX_DIR), "--model", emb_model]
        res = subprocess.run(cmd, capture_output=True, text=True)
        st.code(res.stdout or res.stderr)

st.divider()
st.subheader("4) Ask")
q = st.text_area("Your question", value="Summarize key policy impacts and strategic implications for AI adoption in the industry.")
if st.button("Run RAG"):
    if not (INDEX_DIR / "faiss.index").exists():
        st.error("Index not found. Please run Ingest and Build Index first.")
    else:
        cmd = ["python", "src/rag.py", "--index_dir", str(INDEX_DIR), "--emb_model", emb_model, "--mode", mode, "--top_k", str(top_k), "--query", q]
        res = subprocess.run(cmd, capture_output=True, text=True)
        st.markdown(res.stdout if res.stdout else f"```\n{res.stderr}\n```")
