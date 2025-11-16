# DSA4213 - Oncology Retrieval-Augmented Generation (RAG) Chatbot 
This repository contains the full implementation for our DSA4213 project evaluating Retrieval-Augmented Generation (RAG) for clinical question answering in oncology. The study compares RAG versus a no-retrieval baseline and includes ablations over retriever type, number of retrieved passages 𝐾, and knowledge base design.

## Project Overview 
Clinical question answering is a high-stakes setting where factual accuracy and verifiability matter. While RAG is often proposed as a solution to hallucinations, its behaviour under different retrieval configurations is not well understood.
Our project systematically evaluates:
- **RAG vs No-RAG**
- **Retrieval methods:** FAISS (dense) vs BM25 (sparse)
- **Number of retrieved documents:** K ∈ {3, 5, 7}
- **Knowledge bases:** PubMed corpus vs Cancer.gov QA corpus

Experiments were conducted on **729 Cancer.gov QA pairs** (MedQuAD dataset).

## Setup
This project uses **`pyproject.toml` + `Poetry`** for dependency management.
1. Install uv.
```bash
pip install poetry
```
2. Install dependencies.
```bash
poetry install
```
3. Add your OpenAI API key
```bash
cp .env.sample .env
```
Then edit `.env`:
```bash
OPENAI_API_KEY=your_openai_api_key_here
```
4. Running any script
```bash
poetry run python src/<script_name>.py
```

## Repository Structure
```text
├── 1_CancerGov_QA/            # Original Cancer.gov XML files (MedQuAD)
├── data/                      # Processed corpus (PubMed / QA KB v1)
├── data_v2/                   # Alternative processed corpus (v2)
├── store/                     # FAISS / BM25 indices for v1
├── store_v2/                  # FAISS / BM25 indices for v2
├── src/                       # Main code for retrieval, RAG, and evaluation
├── results/                   # Evaluation outputs (CSV, plots)
├── logs/                      # Log files from experiments
├── CancerGov_QA_Dataset.csv   # 729 Cancer QA pairs used for evaluation
└── README.md
```

## How to run main experiment
1. Build/update vector stores
```bash
poetry run python src/build_faiss_store.py
poetry run python src/build_bm25_store.py     # if using BM25
```
2. Run the baseline and RAG evaluation

The main settings are in src/config.py.
To switch between RAG and No-RAG:
- Set ENABLE_RAG = True for RAG-enabled runs
- Set ENABLE_RAG = False for the No-RAG baseline

(You can also adjust RETRIEVER_TYPE, K, knowledge base, etc. in this file.)

3. Run the evaluation
```bash
poetry run python src/evaluate_rag.py
```

