from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Data Querying
TERM = "cancer"
LIMIT = 50
OUT_DIR = BASE_DIR / "data"

ENABLE_RAG = False
K = 3
