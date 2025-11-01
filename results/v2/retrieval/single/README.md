# Retrieval Performance Comparison (K=3)

- Question-Answer pairs are taken from MedQuad dataset, subset on 1_CancerGov_QA.
- Document corpus is constructed from the url links provided in the MedQuad dataset.

| Metric                | SapBERT | ClinicalBERT | MedEmbed-base | PubMedBERT-base | OpenAI Small | BM25  |
| :-------------------- | :------ | :----------- | :------------ | :-------------- | ------------ | ----- |
| **Exact Match Rate**  | 0.598   | 0.104        | 0.554         | 0.539           | **0.630**    | 0.551 |
| **Average Precision** | 0.746   | 0.300        | 0.713         | 0.692           | **0.765**    | 0.706 |
| **Average Recall**    | 0.920   | 0.575        | 0.900         | 0.877           | **0.922**    | 0.890 |
| **Average F1-Score**  | 0.800   | 0.378        | 0.771         | 0.749           | **0.813**    | 0.763 |
