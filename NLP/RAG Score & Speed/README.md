
# RAG Score Evaluation & Speed Analysis

**Goal**  
Benchmark retrieval quality (0–5 rubric) and speed across backends; compare category-aware vs global.

**Backends**
- SBERT (all-MiniLM-L6-v2) + ScaNN
- T5-base encoder + FAISS
- TF-IDF + Annoy (angular)
- USE (TF-Hub) + HNSW (hnswlib)
- Word2Vec

**Scoring & Metrics**
- Similarity (0–5) via Gemini 1.5 Flash (Vertex AI)
- Average retrieval time per query
- (Planned) classic P@K/R@K/MRR/nDCG

## How to Run
- `notebooks/RAG Score Evaluation and Speed Analysis.ipynb`
- `pip install` per README root; set Vertex AI project/region if using LLM scoring.

## Results
- Suggested results table in notebook; export CSVs to `reports/`.
