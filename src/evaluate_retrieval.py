"""
evaluate_retrieval.py
- Retrieval evaluation script for the Medical RAG system
- Checks how many questions can retrieve the relevant files shown in 'references' field
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

# Configuration
FAISS_PATH = "../store/faiss/openai-text-embedding-3-small"
DATA_PATH = "../data/plain"
EVALUATE_DATA_PATH = "../data/evaluation/retrieval/single.json"
RESULTS_PATH = "../results/retrieval/single/openai-text-embedding-3-small.json"
K = 3

# embedding_model = HuggingFaceEmbeddings(
#     model_name="emilyalsentzer/Bio_ClinicalBERT"
# )
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

# Initialize FAISS database
db = FAISS.load_local(FAISS_PATH, embedding_model,
                      allow_dangerous_deserialization=True)


def load_evaluation_data() -> Dict[str, Any]:
    """Load evaluation data from JSON file"""
    with open(EVALUATE_DATA_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def retrieve_documents(question: str, k: int = 3) -> List[Document]:
    """Retrieve relevant documents for the question"""
    return db.similarity_search(question, k=k)


def extract_source_filename(metadata: Dict[str, Any]) -> str:
    """Extract the filename from document metadata"""
    source = metadata.get("source", "")
    if source:
        # Extract filename from path
        filename = os.path.basename(source)
        return filename
    return ""


def evaluate_retrieval_for_question(question_data: Dict[str, Any], k: int = 3) -> Dict[str, Any]:
    """Evaluate retrieval for a single question"""
    question_id = question_data["id"]
    question = question_data["question"]
    expected_files = set(question_data.get("reference", []))

    # Retrieve documents
    retrieved_docs = retrieve_documents(question, k=k)

    # Extract source filenames from retrieved documents
    retrieved_files = set()
    retrieved_sources = []

    for doc in retrieved_docs:
        filename = extract_source_filename(doc.metadata)
        if filename:
            retrieved_files.add(filename)
            retrieved_sources.append({
                "filename": filename,
                "chunk_id": doc.metadata.get("id", "")
            })

    # Calculate metrics
    true_positives = len(expected_files.intersection(retrieved_files))
    false_positives = len(retrieved_files - expected_files)
    false_negatives = len(expected_files - retrieved_files)

    # Precision: TP / (TP + FP)
    precision = true_positives / len(retrieved_files) if len(retrieved_files) > 0 else 0.0

    # Recall: TP / (TP + FN)
    recall = true_positives / len(expected_files) if len(expected_files) > 0 else 0.0

    # F1 Score: 2 * (precision * recall) / (precision + recall)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Exact match: all expected files retrieved and no extra files
    exact_match = (retrieved_files == expected_files)

    # At least one correct: at least one expected file was retrieved
    at_least_one_correct = true_positives > 0

    return {
        "question_id": question_id,
        "question": question,
        "expected_files": list(expected_files),
        "retrieved_files": list(retrieved_files),
        "retrieved_sources": retrieved_sources,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "exact_match": exact_match,
        "at_least_one_correct": at_least_one_correct,
        "num_expected": len(expected_files),
        "num_retrieved": len(retrieved_files)
    }


def calculate_overall_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate overall retrieval performance metrics"""
    if not results:
        return {}

    # Aggregate metrics
    total_questions = len(results)
    exact_matches = sum(1 for r in results if r["exact_match"])
    at_least_one_correct = sum(1 for r in results if r["at_least_one_correct"])

    # Average metrics
    avg_precision = sum(r["precision"] for r in results) / total_questions
    avg_recall = sum(r["recall"] for r in results) / total_questions
    avg_f1_score = sum(r["f1_score"] for r in results) / total_questions

    # Micro-averaged metrics (aggregate TP, FP, FN first, then calculate)
    total_tp = sum(r["true_positives"] for r in results)
    total_fp = sum(r["false_positives"] for r in results)
    total_fn = sum(r["false_negatives"] for r in results)

    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0

    return {
        "total_questions": total_questions,
        "exact_matches": exact_matches,
        "exact_match_rate": exact_matches / total_questions,
        "at_least_one_correct": at_least_one_correct,
        "at_least_one_correct_rate": at_least_one_correct / total_questions,
        "average_precision": avg_precision,
        "average_recall": avg_recall,
        "average_f1_score": avg_f1_score,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1_score": micro_f1,
        "total_true_positives": total_tp,
        "total_false_positives": total_fp,
        "total_false_negatives": total_fn,
    }


def main():
    print("Loading evaluation data...")
    eval_data = load_evaluation_data()
    questions = eval_data["questions"]

    print(f"Evaluating retrieval for {len(questions)} questions...")

    results = {
        "metadata": {
            "total_questions": len(questions),
            "evaluation_timestamp": datetime.now().isoformat(),
            "retrieval_k": K,
            "database_path": FAISS_PATH,
            "evaluation_data": EVALUATE_DATA_PATH
        },
        "detailed_results": [],
        "summary": {}
    }

    detailed_results = []

    for i, question_data in enumerate(questions, 1):
        print(
            f"Processing question {i}/{len(questions)}: {question_data['id']}")

        try:
            result = evaluate_retrieval_for_question(question_data, K)
            detailed_results.append(result)
        except Exception as e:
            print(
                f"  Error processing question {question_data['id']}: {str(e)}")
            # Add error result
            error_result = {
                "question_id": question_data["id"],
                "question": question_data["question"],
                "error": str(e),
                "expected_files": question_data.get("reference", []),
                "retrieved_files": [],
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "exact_match": False,
                "at_least_one_correct": False
            }
            detailed_results.append(error_result)

    # Calculate overall metrics
    results["detailed_results"] = detailed_results
    results["summary"] = calculate_overall_metrics(detailed_results)

    # Save results
    with open(RESULTS_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Detailed results saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
