# B. RAG Score Evaluation & Speed Analysis

This notebook benchmarks retrieval quality and speed for several retrieval backends in a RAG pipeline. It also compares category-aware retrieval vs global retrieval on a Q&A dataset.

## Dataset
Format: JSONL with one record per Q&A:


Example filename: output_with_categories_1107.jsonl.

## Retrieval Backends Evaluated
- SBERT (all-MiniLM-L6-v2) + ScaNN
- T5-base encoder embeddings + FAISS (encoder hidden-state mean pooling → vectors)
- TF-IDF + Annoy (angular ≈ cosine)
- Universal Sentence Encoder (TF-Hub) + HNSW (hnswlib)
- Word2Vec implementation

Each backend is evaluated with category and without category to study the effect of filtering/boosting by category.

## Scoring & Metrics
- Similarity Score (0–5): For each query’s Top-5 neighbors, a Vertex AI (Gemini 1.5 Flash) prompt assigns a 0–5 similarity score using a structured rubric.
- Speed: Average retrieval time per query (seconds).
- Query Set: Random 20 queries sampled from the dataset.

Note: The notebook prints Average Similarity Score and Average Retrieval Time per model/setting. Classic IR metrics (Precision@K, Recall@K, MRR, nDCG) are not computed yet and are listed in the roadmap.

## Environment & Dependencies
Install the libraries used across the experiments:
```bash
pip install \
  sentence-transformers scann faiss-cpu transformers torch \
  scikit-learn annoy jsonlines numpy \
  tensorflow tensorflow-hub hnswlib \
  google-cloud-aiplatform vertexai
