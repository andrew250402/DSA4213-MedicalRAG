from pathlib import Path
import json, re, argparse

def chunk_tokens(text, size=250, overlap=40):
    toks = re.findall(r"\S+", text)
    step = max(1, size - overlap)
    for i in range(0, max(1, len(toks) - size + 1), step):
        yield " ".join(toks[i:i+size])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="DSA4213-MedicalRAG/Adrian/data")
    ap.add_argument("--out", default="artifacts/chunks.jsonl")
    ap.add_argument("--size", type=int, default=250)
    ap.add_argument("--overlap", type=int, default=40)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(Path(args.data_dir).glob("*.plain.txt"))
    with out.open("w", encoding="utf-8") as f:
        for tf in files:
            base = tf.stem.replace(".plain", "")
            txt = tf.read_text(encoding="utf-8", errors="ignore").strip()
            if not txt:
                continue
            wrote = False
            for idx, passage in enumerate(chunk_tokens(txt, args.size, args.overlap), 1):
                f.write(json.dumps({"doc_id": base, "chunk_id": f"{base}_{idx}", "text": passage}, ensure_ascii=False) + "\n")
                wrote = True
            if not wrote:
                f.write(json.dumps({"doc_id": base, "chunk_id": f"{base}_1", "text": txt}, ensure_ascii=False) + "\n")
    print("Wrote:", out, "— lines:", sum(1 for _ in open(out, "r", encoding="utf-8")))

if __name__ == "__main__":
    main()
