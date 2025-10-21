"""
ingest_data.py
- Loads raw medical data (txt, pdf, etc.)
- Cleans text (remove headers, references, etc.)
- Splits into chunks for embeddings
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import os


def load_text_files(data_dir="data/plain/"):
    docs = []
    for file in os.listdir(data_dir):
        if file.endswith(".txt"):
            with open(os.path.join(data_dir, file), "r", encoding="utf-8") as f:
                docs.append(f.read())
    return docs


def split_documents(texts, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return [Document(page_content=chunk) for text in texts for chunk in splitter.split_text(text)]


if __name__ == "__main__":
    texts = load_text_files()
    chunks = split_documents(texts)
    print(f"Loaded {len(chunks)} chunks")
