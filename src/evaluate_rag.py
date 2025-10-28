import os
import pandas as pd
from tqdm import tqdm
import logging
from datetime import datetime
from rag_pipeline import build_agent
import config

from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.metrics import f1_score
import nltk
import re

# Ensure nltk tokenizer is available
nltk.download("punkt", quiet=True)

# =====================
# LOGGING SETUP
# =====================
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = os.path.join(LOG_DIR, f"rag_eval_metrics_{timestamp}.log")

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)

# =====================
# HELPER FUNCTIONS
# =====================

def preprocess_text(text: str):
    """Lowercase and remove punctuation for fairer comparison."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text

def compute_f1(pred, ref):
    """Token-level F1."""
    pred_tokens = preprocess_text(pred).split()
    ref_tokens = preprocess_text(ref).split()
    common = set(pred_tokens) & set(ref_tokens)
    if not pred_tokens or not ref_tokens:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

# =====================
# LOG CONFIG DETAILS
# =====================
logging.info("========== RAG EVALUATION START ==========")
logging.info("Experiment Configuration:")
logging.info(f"• ENABLE_RAG       = {getattr(config, 'ENABLE_RAG', True)}")
logging.info(f"• RETRIEVER_TYPE   = {getattr(config, 'RETRIEVER_TYPE', 'faiss')}")
logging.info(f"• MODEL            = {getattr(config, 'MODEL', 'gpt-3.5-turbo')}")
logging.info(f"• K                = {getattr(config, 'K', 3)}")
logging.info(f"• PROMPT_STYLE     = {getattr(config, 'PROMPT_STYLE', 'citation')}")
logging.info(f"• FAISS_PATH       = {getattr(config, 'FAISS_PATH', 'N/A')}")
logging.info(f"• BM25_PATH        = {getattr(config, 'BM25_PATH', 'N/A')}")
logging.info("==========================================\n")

# =====================
# LOAD DATASET
# =====================
try:
    df = pd.read_csv("CancerGov_QA_Dataset.csv").dropna(subset=["question", "answer"])
    logging.info(f"Loaded {len(df)} QA pairs for evaluation.")
except FileNotFoundError:
    logging.error("Dataset file 'CancerGov_QA_Dataset.csv' not found.")
    raise

# =====================
# INITIALIZE AGENT
# =====================
try:
    agent = build_agent()
    logging.info("Successfully initialized RAG agent.\n")
except Exception as e:
    logging.exception("Failed to initialize RAG agent.")
    raise

# =====================
# METRIC SETUP
# =====================
rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
smooth_fn = SmoothingFunction().method1

results = []
rouge_scores, bleu_scores, f1_scores = [], [], []

# =====================
# EVALUATION LOOP
# =====================
for i, row in tqdm(df.iterrows(), total=len(df)):
    question = row["question"]
    ground_truth = row["answer"]

    try:
        # Run RAG model
        response = ""
        for step in agent.stream(
            {"messages": [{"role": "user", "content": question}]},
            stream_mode="values",
        ):
            response = step["messages"][-1].content  # last assistant message

        # ---- Compute metrics ----
        rougeL = rouge.score(ground_truth, response)["rougeL"].fmeasure
        bleu = sentence_bleu(
            [preprocess_text(ground_truth).split()],
            preprocess_text(response).split(),
            smoothing_function=smooth_fn
        )
        f1 = compute_f1(response, ground_truth)

        results.append({
            "question": question,
            "predicted_answer": response,
            "ground_truth": ground_truth,
            "rougeL": rougeL,
            "bleu": bleu,
            "f1": f1
        })

        rouge_scores.append(rougeL)
        bleu_scores.append(bleu)
        f1_scores.append(f1)

        logging.info(f"\n=== QA {i+1}/{len(df)} ===")
        logging.info(f"Question: {question}")
        logging.info(f"Predicted: {response}")
        logging.info(f"Ground Truth: {ground_truth}")
        logging.info(f"ROUGE-L: {rougeL:.3f}, BLEU: {bleu:.3f}, F1: {f1:.3f}")

    except Exception as e:
        logging.exception(f"Error evaluating question index {i}: {e}")

# =====================
# SAVE RESULTS
# =====================
eval_df = pd.DataFrame(results)
eval_output = "rag_evaluation_results_metrics.csv"
eval_df.to_csv(eval_output, index=False)

# =====================
# SUMMARY
# =====================
if rouge_scores:
    avg_rouge = sum(rouge_scores) / len(rouge_scores)
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    avg_f1 = sum(f1_scores) / len(f1_scores)
    logging.info("\n===== METRIC SUMMARY =====")
    logging.info(f"Average ROUGE-L: {avg_rouge:.3f}")
    logging.info(f"Average BLEU:    {avg_bleu:.3f}")
    logging.info(f"Average F1:      {avg_f1:.3f}")
else:
    logging.warning("No metrics computed — check your results.")

logging.info("\nSaved evaluation results → rag_evaluation_results_metrics.csv")
logging.info(f"Log file saved at: {LOG_FILE}")
logging.info("==========================================\n")
