"""
build_vectorstore.py
- Embeds documents
- Stores them in FAISS or Chroma
"""

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import pickle
from ingest_data import load_text_files, split_documents

def build_vector_db():
    # Load + chunk docs
    texts = load_text_files()
    chunks = split_documents(texts)

    # ✅ Correct model + package
    embedding_model = HuggingFaceEmbeddings(
        model_name="cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    )

    # Build FAISS index
    db = FAISS.from_documents(chunks, embedding_model)

    # Save index to disk
    with open("vectorstore.pkl", "wb") as f:
        pickle.dump(db, f)

    return db

if __name__ == "__main__":
    db = build_vector_db()
    print("Vector DB built and saved.")
