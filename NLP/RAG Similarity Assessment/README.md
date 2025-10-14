
# Similarity Assessment Model

**Features**
- Embeddings: SBERT/USE/T5/OpenAI/Cohere/Gemini (pluggable)
- ANN: FAISS / ScaNN / Annoy / HNSW
- Cosine similarity; easy to extend to LLM scoring

**Dataset**
CSV/JSON/JSONL (Q&A or sentence pairs)

## How to Run
- `notebooks/Similarity Assessment Model.ipynb`
- Steps: preprocess → embed → index → retrieve Top-K → score → (optional) export plots

## Results
- Example Top-K neighbors in console, plots saved to `results/`.
