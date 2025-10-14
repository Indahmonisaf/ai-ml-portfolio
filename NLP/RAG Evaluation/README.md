### RAG Evaluation – Category-Aware Question Retrieval (RAG Evaluation.ipynb)
This project evaluates retrieval quality for a Retrieval-Augmented Generation (RAG) system on a Q&A dataset. It compares global retrieval (no category) vs category-aware retrieval (filter/boost by category) and experiments with multiple text representations and ANN indexes.

**Dataset**  
Format: JSONL with fields:  
`{"question": "...", "answer": "...", "category": "..."}`  
Example filename: `output_with_categories_1107.jsonl`.

## Approaches
### Representations
- TF-IDF + cosine similarity (lexical baseline)
- Sentence embeddings via sentence-transformers (semantic)
- Word2Vec (distributional baseline)

### ANN Index
- ScaNN (and Faiss is referenced in code for future swaps)

### Pre-processing
- Janome tokenizer noted for Japanese text support (multilingual-ready)

## Evaluation Protocol
- Hold out ~20 questions as queries.
- Retrieve Top-5 most similar questions from the corpus.
- Compare Without Category vs With Category to observe changes in neighborhood relevance.
- Current notebook prints Top-K neighbors for qualitative inspection.
- Planned metrics: Precision@K, Recall@K, MRR, nDCG.

## Example (qualitative)
Query: “Is credit card payment available?”  
Top-5 (sample):
- “Do you accept credit card payment?”
- “Do you accept credit card payments?”
- “What payment methods are available?”
- “What are the payment options?”
(Shown for illustration; actual neighbors depend on the index/model and dataset.)

## How to Run
Install dependencies:
- sentence-transformers, scann, jsonlines, numpy, scikit-learn (for TF-IDF), optional faiss-cpu, and janome if needed.

Place your dataset at the project root (e.g., output_with_categories_1107.jsonl).

Open **RAG Evaluation.ipynb** and run cells in order:
- Load data → Build representations → Index → Retrieve Top-K → Compare With/Without Category.

(Optional) Add metric computation cells for P@K, R@K, MRR, nDCG and export results to CSV.

## Findings (current state)
The notebook demonstrates the setup and comparison path for category-aware retrieval vs global retrieval.

It prints Top-5 neighbors for manual inspection; quantitative metrics are a recommended next step.

## Roadmap
- Implement Precision@K / Recall@K / MRR / nDCG
- Add BM25 baseline and hybrid search (BM25 + embeddings)
- Benchmark Faiss vs ScaNN (latency, memory, recall)
- Add category-aware re-ranking and ablation studies
- Log & visualize results

## Tech Stack
Python, Jupyter Notebook

sentence-transformers, scann (ANN), jsonlines, numpy, scikit-learn, (optional) faiss-cpu, janome

## Author
Built by Indah Monisa Firdiantika (M.S.) — focused on practical RAG evaluation and category-aware retrieval.
