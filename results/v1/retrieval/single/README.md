# Retrieval Performance Comparison (K=3)

- Document corpus is constructed from a subset of PubMed open-access articles about cancer.
- Question-Answer pairs are generated from an LLM (Gemini 2.5 Pro) based on the document corpus.

| Metric                | SapBERT | ClinicalBERT | MedEmbed-base | PubMedBERT-base | OpenAI Small | BM25      |
| :-------------------- | :------ | :----------- | :------------ | :-------------- | ------------ | --------- |
| **Exact Match Rate**  | 0.40    | 0.08         | 0.520         | 0.480           | **0.58**     | 0.40      |
| **Average Precision** | 0.613   | 0.223        | 0.700         | 0.653           | **0.750**    | 0.640     |
| **Average Recall**    | 0.880   | 0.440        | 0.920         | 0.880           | 0.960        | **0.980** |
| **Average F1-Score**  | 0.693   | 0.283        | 0.767         | 0.720           | **0.813**    | 0.737     |
