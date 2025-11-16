#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def detect_metric_cols(df: pd.DataFrame):
    wanted_order = ["f1", "rouge", "bleu"]
    found = []
    for c in df.columns:
        lc = c.lower()
        if lc in ("f1", "f1_score"):
            found.append(c)
        elif lc.startswith("rouge"):  # matches rougeL, rouge_l, rouge, rouge1, etc.
            found.append(c)
        elif lc in ("bleu", "bleu_score"):
            found.append(c)
    # de-dup preserving first occurrence
    dedup = []
    for c in found:
        if c not in dedup:
            dedup.append(c)
    # reorder: f1 first, then any rouge*, then bleu*
    ordered = []
    # f1
    for c in dedup:
        if c.lower() in ("f1","f1_score"):
            ordered.append(c)
    # rouge
    for c in dedup:
        if c.lower().startswith("rouge"):
            ordered.append(c)
    # bleu
    for c in dedup:
        if c.lower() in ("bleu","bleu_score"):
            ordered.append(c)
    return ordered

def load_with_label(path: Path, label: str):
    df = pd.read_csv(path)
    # If a mode/variant column exists, prefer that; otherwise use provided label
    mode_col = None
    for c in df.columns:
        if c.lower() in {"mode", "variant", "with_rag", "rag"}:
            mode_col = c
            break
    if mode_col is None:
        df["label"] = label
    else:
        df["label"] = df[mode_col].astype(str)
    return df

def main():
    ap = argparse.ArgumentParser(description="Plot evaluation metrics from CSV")
    ap.add_argument("--input", 
                    default=r"c:\Users\Morep\OneDrive\Documents\DSA4213-MedicalRAG\rag_evaluation_results_metrics.csv",
                    help="Path to CSV (default: rag_evaluation_results_metrics.csv)")
    ap.add_argument("--label", default="RAG", help="Label for --input if CSV lacks a mode column")
    ap.add_argument("--compare", help="Optional path to a second CSV (e.g., no_rag CSV)")
    ap.add_argument("--compare-label", default="No-RAG", help="Label for --compare")
    ap.add_argument("--outdir", default=str(Path("figures")), help="Output directory for plots")
    args = ap.parse_args()

    in_path = Path(args.input)
    df1 = load_with_label(in_path, args.label)

    dfs = [df1]
    if args.compare:
        df2 = load_with_label(Path(args.compare), args.compare_label)
        dfs.append(df2)

    df = pd.concat(dfs, ignore_index=True)

    # Detect metric columns
    metric_cols = detect_metric_cols(df)
    if not metric_cols:
        raise SystemExit("No metric columns detected. Ensure your CSV has f1/rouge/bleu or numeric metric columns.")

    # Coerce metrics to numeric
    for c in metric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Barplot of mean metrics per label
    summary = (
        df.groupby("label")[metric_cols]
        .mean(numeric_only=True)
        .reset_index()
        .melt(id_vars="label", var_name="metric", value_name="value")
    )
    plt.figure(figsize=(8, 4))
    sns.barplot(data=summary, x="metric", y="value", hue="label")
    plt.title("Mean metrics per label")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(outdir / "metrics_bar_means.png", dpi=200)
    plt.close()

    # 2) Boxplots (distribution per metric)
    df_melt = df.melt(id_vars=["label"], value_vars=metric_cols, var_name="metric", value_name="value")
    plt.figure(figsize=(10, 4))
    sns.boxplot(data=df_melt, x="metric", y="value", hue="label")
    plt.title("Metric distributions")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(outdir / "metrics_boxplots.png", dpi=200)
    plt.close()

    # 3) Scatter ROUGE vs BLEU if both exist
    def col_like(names):
        for n in names:
            for c in metric_cols:
                lc = c.lower()
                if n == "rouge_any" and lc.startswith("rouge"):
                    return c
                if lc == n:
                    return c
        return None

    rouge_col = col_like(["rouge_l", "rougel", "rouge", "rouge_any"])
    bleu_col = col_like(["bleu", "bleu_score"])
    f1_col = col_like(["f1", "f1_score"])

    if rouge_col and bleu_col:
        plt.figure(figsize=(5, 5))
        sns.scatterplot(data=df, x=rouge_col, y=bleu_col, hue="label", alpha=0.6)
        plt.title(f"{rouge_col} vs {bleu_col}")
        plt.tight_layout()
        plt.savefig(outdir / "scatter_rouge_vs_bleu.png", dpi=200)
        plt.close()

    # 4) Optional: F1 distribution
    if f1_col:
        plt.figure(figsize=(6, 4))
        sns.kdeplot(data=df, x=f1_col, hue="label", fill=True, common_norm=False, alpha=0.4)
        plt.title(f"{f1_col} distribution")
        plt.tight_layout()
        plt.savefig(outdir / "f1_distribution.png", dpi=200)
        plt.close()

    print(f"Saved plots to: {outdir.resolve()}")

if __name__ == "__main__":
    main()