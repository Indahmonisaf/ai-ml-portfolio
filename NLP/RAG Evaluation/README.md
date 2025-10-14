
# RAG Evaluation – Category-Aware Question Retrieval

**Goal**  
Compare global retrieval vs category-aware retrieval on a Q&A dataset.

**Dataset**  
JSONL: `{"question": "...", "answer": "...", "category": "..."}` (e.g., `output_with_categories_1107.jsonl`).

**Approach**
- Representations: TF-IDF + cosine; Sentence-Transformers; Word2Vec
- ANN: ScaNN (Faiss planned)
- Tokenizer note: Janome for Japanese (multilingual)

**Protocol**
- Hold out ~20 queries → retrieve Top-5
- Compare with/without category filter/boost
- Add metrics next (P@K, R@K, MRR, nDCG)

## How to Run
- `notebooks/RAG Evaluation.ipynb` (end-to-end cells)
- Install: sentence-transformers, scann, jsonlines, numpy, scikit-learn, (optional) faiss-cpu, janome

## Results
- Printed neighbors for qualitative check; add metric CSVs later.
