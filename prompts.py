from .retrieval import make_ctx

def prompt_plain(question, ctx):
    return f"""Question: {question}

Context:
{make_ctx(ctx)}

Answer in ≤120 words. Be concise and neutral."""

def prompt_citation(question, ctx):
    return f"""You are an evidence-first oncology assistant. Use only the context.

Question:
{question}

Context:
{make_ctx(ctx)}

Instructions:
- ≤120 words.
- Make only claims supported by the context.
- Add numbered citations [i] after the supporting claims.
- If evidence is insufficient or conflicting, say so.
- This is not medical advice.

Answer:"""

def prompt_evidence_then_answer(question, ctx):
    return f"""Ground every claim in the context.

Question:
{question}

Context:
{make_ctx(ctx)}

Write:
1) Key Evidence — 2–4 bullets (≤20 words each, include a short quote + [i])
2) Answer — 2–3 sentences with citations [i]. ≤120 words. Not medical advice.

Answer:"""

def prompt_safety_deferral(question, ctx):
    return f"""You are an evidence-first oncology assistant. Use only the context.

Question:
{question}

Context:
{make_ctx(ctx)}

Instructions:
- ≤120 words.
- Only make claims supported by the context; add [i] after supporting claims.
- If the question implies red-flag symptoms/acute deterioration, advise urgent care.
- If evidence is insufficient or conflicting, say so.
- This is not medical advice.

Answer:"""

FEWSHOT = """Example
Q: When should colorectal cancer screening start?
Context:
[1] Screening programs detect polyps and early-stage tumors.
[2] Colonoscopy is commonly used in organized programs; intervals vary by risk.
A: Average-risk adults should take part in population screening; options include stool tests or colonoscopy [1][2].
"""

def prompt_fewshot(question, ctx):
    return f"""You are an evidence-first oncology assistant. Use only the context.

{FEWSHOT}
Now answer the new question in the same style.

Question:
{question}

Context:
{make_ctx(ctx)}

Write 2–3 sentences, each with citations [i]. ≤120 words. Not medical advice.

Answer:"""

def prompt_gold_style(q, ctx):
    return f"""Answer the medical question in ONE short sentence (≤20 words).
Use ONLY the context and DO NOT add citations or disclaimers.

Question:
{q}

Context:
{make_ctx(ctx)}

Answer:"""

PROMPTS = {
    "plain":     prompt_plain,
    "citation":  prompt_citation,
    "evidence":  prompt_evidence_then_answer,
    "safety":    prompt_safety_deferral,
    "fewshot":   prompt_fewshot,
    "gold style": prompt_gold_style,
}

def build_prompt(style, question, ctx):
    return PROMPTS[style](question, ctx)
