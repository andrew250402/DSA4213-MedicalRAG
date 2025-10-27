import os
import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path

# ===== CONFIG =====
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / "1_CancerGov_QA"
OUTPUT_FILE = BASE_DIR / "CancerGov_QA_Dataset.csv"

def parse_xml_file(file_path):
    """Parse a single CancerGov XML file and extract QA pairs."""
    tree = ET.parse(file_path)
    root = tree.getroot()

    doc_id = root.attrib.get("id", "")
    source = root.attrib.get("source", "")
    url = root.attrib.get("url", "")
    focus = root.findtext("Focus", "").strip()

    qa_pairs = []

    qa_parent = root.find("QAPairs")
    if qa_parent is None:
        return []

    for pair in qa_parent.findall("QAPair"):
        pid = pair.attrib.get("pid", "")

        q_node = pair.find("Question")
        a_node = pair.find("Answer")

        if q_node is None or a_node is None:
            continue

        qid = q_node.attrib.get("qid", "")
        qtype = q_node.attrib.get("qtype", "")
        question = (q_node.text or "").strip().replace("\n", " ")
        answer = (a_node.text or "").strip().replace("\n", " ")

        qa_pairs.append({
            "document_id": doc_id,
            "source": source,
            "url": url,
            "focus": focus,
            "pid": pid,
            "question_id": qid,
            "question_type": qtype,
            "question": question,
            "answer": answer,
        })

    return qa_pairs

def convert_all_xmls(input_dir=INPUT_DIR, output_file=OUTPUT_FILE):
    """Convert all XML files in the folder into a single CSV dataset."""
    all_records = []

    xml_files = [f for f in os.listdir(input_dir) if f.endswith(".xml")]
    print(f"[INFO] Found {len(xml_files)} XML files in {input_dir}")

    for xml_file in xml_files:
        file_path = input_dir / xml_file
        try:
            records = parse_xml_file(file_path)
            all_records.extend(records)
            print(f"[OK] Parsed {xml_file} ({len(records)} QA pairs)")
        except Exception as e:
            print(f"[ERROR] Failed to parse {xml_file}: {e}")

    if not all_records:
        print("[WARN] No QA pairs found. Check your input directory.")
        return

    df = pd.DataFrame(all_records)
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"[SUCCESS] Saved dataset with {len(df)} rows → {output_file}")

if __name__ == "__main__":
    convert_all_xmls()
