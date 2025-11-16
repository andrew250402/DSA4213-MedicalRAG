#!/usr/bin/env python3
"""
Script to generate evaluation data in JSON format from CancerGov_QA_Dataset.csv
The output format matches the structure of /data/evaluation/retrieval/single.json
"""

import csv
import json
from pathlib import Path


def main():
    # Input and output paths
    csv_path = Path("CancerGov_QA_Dataset.csv")
    output_path = Path("data_v2/evaluation/retrieval/single.json")
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Read CSV and generate questions
    questions = []
    question_counter = 1
    
    with open(csv_path, 'r', encoding='utf-8-sig') as csvfile:
        # utf-8-sig automatically removes BOM
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            # Extract document_id and construct reference filename
            document_id = row['document_id']
            reference_file = f"{document_id}.txt"
            
            # Create question entry
            question_entry = {
                "id": question_counter,
                "question": row['question'],
                "answer": row['answer'],
                "reference": [reference_file]
            }
            
            questions.append(question_entry)
            question_counter += 1
    
    # Create the output structure
    output_data = {
        "version": "v2.0",
        "description": f"{len(questions)} questions from CancerGov_QA_Dataset.csv with references mapped to document IDs.",
        "questions": questions
    }
    
    # Write to JSON file
    with open(output_path, 'w', encoding='utf-8') as jsonfile:
        json.dump(output_data, jsonfile, indent=4, ensure_ascii=False)
    
    print(f"Successfully generated {output_path}")
    print(f"Total questions: {len(questions)}")


if __name__ == "__main__":
    main()
