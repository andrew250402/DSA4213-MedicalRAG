import re, json
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import wilcoxon

from retrieval import load_chunks, build_bm25, retrieve
from prompts   import build_prompt
from generate  import load_model, call_model

from collections import Counter

def _tok(s): 
    return re.findall(r"\w+", (s or "").lower())

def rouge_l_f1(pred, ref):
    p, r = _tok(pred), _tok(ref)
    if not p or not r: return 0.0
    m, n = len(p), len(r)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m):
        for j in range(n):
            dp[i+1][j+1] = dp[i][j]+1 if p[i]==r[j] else max(dp[i][j+1], dp[i+1][j])
    lcs = dp[m][n]
    prec = lcs / m
    rec  = lcs / n
    return 0.0 if (prec+rec)==0 else 2*prec*rec/(prec+rec)

def token_f1(pred, ref):
    p, r = _tok(pred), _tok(ref)
    if not p or not r: return np.nan
    cp, cr = Counter(p), Counter(r)
    overlap = sum((cp & cr).values())
    prec = overlap / max(1, sum(cp.values()))
    rec  = overlap / max(1, sum(cr.values()))
    return 0.0 if (prec+rec)==0 else 2*prec*rec/(prec+rec)

def citation_rate(ans):
    sents = re.split(r'(?<=[.!?])\s+', ans or "")
    return 0.0 if not sents else sum(1 for s in sents if re.search(r"\[\d+\]", s)) / len(sents)

def faithfulness_bigram(ans, ctx_text):
    a = ans.split(); big = set(zip(a, a[1:])) if len(a)>1 else set()
    b = ctx_text.split(); bb = set(zip(b, b[1:])) if len(b)>1 else set()
    return len(big & bb) / max(1, len(big)) if big else 0.0

def paired_wilcoxon(df, style_a, style_b, metric="rougeL"):
    a = df[df["style"]==style_a][["question", metric]].dropna()
    b = df[df["style"]==style_b][["question", metric]].dropna()
    both = a.merge(b, on="question", suffixes=("_a","_b"))
    if len(both) < 10:
        return {"pairs": len(both), "stat": None, "p_value": None}
    stat, p = wilcoxon(both[f"{metric}_a"], both[f"{metric}_b"])
    return {"pairs": int(len(both)), "stat": float(stat), "p_value": float(p)}

# medquad loader
def _pick(cols, candidates):
    for c in candidates:
        for name in cols:
            if name.lower()==c.lower(): return name
    for c in candidates:
        for name in cols:
            if c.lower() in name.lower(): return name
    return None

def load_medquad_oncology(path="medquad.csv"):
    mq = pd.read_csv(path)
    focus_col = _pick(mq.columns, ["focus_area","focus area","focus","topic","category"])
    q_col     = _pick(mq.columns, ["question","query","q"])
    a_col     = _pick(mq.columns, ["answer","gold","reference","ref","a"])
    assert all([focus_col, q_col, a_col]), f"Missing cols. Got: {list(mq.columns)}"

    def is_oncology(label):
        if not isinstance(label, str): return False
        s = label.lower()
        return any(t in s for t in [
            "oncolog","cancer","neoplasm","tumor","tumour",
            "carcinoma","sarcoma","leukem","lymphoma","melanom"
        ])

    mq = mq[mq[focus_col].apply(is_oncology)].copy()
    mq = mq.rename(columns={q_col:"gold_question", a_col:"gold_answer"})
    mq = mq.dropna(subset=["gold_question","gold_answer"]).reset_index(drop=True)
    return mq

# main driver
def main(
    chunks_path = "artifacts/chunks.jsonl",
    medquad_csv = "medquad.csv",
    model_id    = "google/flan-t5-base",
    styles      = ("plain","citation","evidence","fewshot","safety","gold style"),
    N           = 100,
    topk        = 3,
    out_csv     = "results/prompt_styles_from_medquad.csv"
):
    # Retrieval
    chunks = load_chunks(chunks_path)
    bm25, ids, texts, tok = build_bm25(chunks)
    cid2text = {c["chunk_id"]: c["text"] for c in chunks}

    # Generator
    tokz, model, device = load_model(model_id)
    print(f"Device: {device} | Model: {model_id}")

    # Eval set
    mq_onc = load_medquad_oncology(medquad_csv)
    N = min(N, len(mq_onc))
    EVAL = mq_onc.sample(N, random_state=0)[["gold_question","gold_answer"]]

    # Run
    rows = []
    for style in styles:
        for _, row in tqdm(EVAL.iterrows(), total=len(EVAL), desc=f"style={style}"):
            q, gold = row["gold_question"], row["gold_answer"]
            ctx = retrieve(bm25, ids, texts, tok, q, k=topk)
            prompt = build_prompt(style, q, ctx)
            ans = call_model(tokz, model, device, prompt, max_new_tokens=120)

            ctx_ids = [c["chunk_id"] for c in ctx]
            ctx_text = " ".join(cid2text.get(cid,"") for cid in ctx_ids)

            rows.append({
                "style": style,
                "question": q,
                "answer": ans,
                "gold_answer": gold,
                "ctx_ids": ctx_ids,
                "len_words": len(ans.split()),
                "rougeL": rouge_l_f1(ans, gold),
                "f1_tok": token_f1(ans, gold),
                "citation_rate": citation_rate(ans),
                "faithfulness_proxy": faithfulness_bigram(ans, ctx_text),
            })

    df = pd.DataFrame(rows)
    Path("results").mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv} (rows={len(df)})")

    # Summaries
    overlap = (
        df.groupby("style")
          .agg(n=("question","count"),
               rougeL_median=("rougeL","median"),
               f1_median=("f1_tok","median"),
               words_median=("len_words","median"))
          .sort_values("rougeL_median", ascending=False)
    )
    print("\n=== Overlap / length ===")
    print(overlap)

    grounding = (
        df.groupby("style")
          .agg(faithfulness_median=("faithfulness_proxy","median"),
               citation_rate_mean=("citation_rate","mean"))
          .sort_values("faithfulness_median", ascending=False)
    )
    print("\n=== Grounding ===")
    print(grounding)

    # Significance vs. plain (ROUGE-L)
    print("\n=== Wilcoxon (ROUGE-L) vs plain ===")
    for s in styles:
        if s == "plain": continue
        print(s, paired_wilcoxon(df, s, "plain", "rougeL"))

if __name__ == "__main__":
    main()
