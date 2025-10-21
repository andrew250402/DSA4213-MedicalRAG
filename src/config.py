from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Data Querying
TERM = "cancer"
LIMIT = 50
OUT_DIR = BASE_DIR / "data"

# Rag Pipeline
ENABLE_RAG = True
K = 3
RETRIEVER_TYPE = "faiss"   # Options: "faiss" or "bm25"

# Paths to vector stores
FAISS_PATH = "store/faiss/openai-text-embedding-3-small"
BM25_PATH = "store/bm25"
