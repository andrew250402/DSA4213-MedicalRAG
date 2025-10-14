from pathlib import Path
import json, re
from rank_bm25 import BM25Okapi

def load_chunks(path="artifacts/chunks.jsonl"):
    return [json.loads(l) for l in open(path, "r", encoding="utf-8")]

def build_bm25(chunks):
    texts = [c["text"] for c in chunks if c.get("text", "").strip()]
    ids   = [c["chunk_id"] for c in chunks if c.get("text", "").strip()]
    assert texts, "No texts loaded — recheck your DATA_DIR or chunking output."
    tok = lambda s: re.findall(r"\w+", s.lower())
    index = BM25Okapi([tok(t) for t in texts])
    return index, ids, texts, tok

def retrieve(index, ids, texts, tok, query, k=3):
    scores = index.get_scores(tok(query))
    top = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]
    return [{"chunk_id": ids[i], "text": texts[i]} for i,_ in top]

def make_ctx(ctx, max_total_chars=1400, per_chunk_chars=450):
    out, used = [], 0
    for i, c in enumerate(ctx, 1):
        s = f"[{i}] {c['text'][:per_chunk_chars]}"
        if used + len(s) > max_total_chars: break
        out.append(s); used += len(s)
    return "\n\n".join(out)
