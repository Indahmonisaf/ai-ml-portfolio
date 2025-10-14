
# Similarity Assessment Model

This project implements a **semantic similarity** pipeline for text pairs and **nearest-neighbor retrieval** for RAG/QA scenarios. It encodes texts into vector representations, indexes them with ANN libraries, and scores/reports the similarity of candidate neighbors.

---

## Notebook
**`Similarity Assessment Model.ipynb`** (Jupyter/Colab)

---

## Features

- **Embeddings:** `sentence-transformers` (e.g., SBERT / `all-MiniLM`), with room to swap in **USE / T5 / OpenAI / Cohere / Gemini** encoders.
- **ANN Indexes:** **FAISS**, **ScaNN**, **Annoy**, **HNSW (hnswlib)** — pluggable backends for speed/recall trade-offs.
- **Similarity:** **Cosine similarity** (baseline), easily extendable to other distance metrics or **LLM-based** scoring.
- **Evaluation:** Prints **Top-K** neighbors and similarity scores for qualitative assessment; ready to add **IR metrics**.

---

## Dataset

- **Expected format:** text records in **CSV / JSON / JSONL** (Q&A or sentence pairs).
- Refer to the *Referenced Data Files* table in the notebook for exact filenames/paths.

---

## Environment & Dependencies

Install common dependencies used in the notebook (adjust as needed):

```bash
pip install \
  numpy pandas scikit-learn \
  sentence-transformers \
  faiss-cpu scann annoy hnswlib \
  jsonlines tqdm
````

**Optional:**

* `faiss-gpu` if you have a compatible GPU.
* `tensorflow` / `tensorflow-hub` or `transformers` if you switch encoders.

---

## How to Run

1. **Place** your dataset files in the project folder.
2. **Open** `Similarity Assessment Model.ipynb` in Jupyter/Colab.
3. **Run cells in order:**

   * **Load & preprocess data**
   * **Build embeddings**
   * **Create ANN index**
   * **Retrieve Top-K neighbors**
   * **Score and inspect results**
4. *(Optional)* **Export** results (CSV) and create **plots** (e.g., similarity distribution, model/index comparisons).

---

## Example Output (conceptual)

**Query:** “How can I reset my account password?”

**Top-5 neighbors (cosine similarity):**

1. “I forgot my password, how do I reset it?”   **0.89**
2. “Steps to change login password”   **0.83**
3. “Account recovery flow if password is lost”   **0.81**
4. “Where to find password reset option”   **0.79**
5. “Update my credentials for sign-in”   **0.76**

---

## Suggested Results Table (fill after running)

| Encoder / Model  | Index | Top-K | Avg Cosine Sim | Notes             |
| ---------------- | :---: | :---: | :------------: | ----------------- |
| all-MiniLM-L6-v2 | FAISS |   5   |        …       | Fast baseline     |
| all-MiniLM-L6-v2 | ScaNN |   5   |        …       | Good recall/speed |
| USE              |  HNSW |   5   |        …       | —                 |
| TF-IDF           | Annoy |   5   |        …       | Lexical baseline  |

---

## Roadmap

* Add **Precision@K / Recall@K / MRR / nDCG**
* Export **CSV** of query–neighbor pairs and **matplotlib** plots
* Benchmark **latency vs. recall** across **FAISS / ScaNN / Annoy / HNSW**
* Try **hybrid search** (BM25/TF-IDF + embeddings)
* Add **category-aware filtering/re-ranking** if your data contains labels

---

## Author

Built by **Indah Monisa Firdiantika (M.S.)** — focusing on robust, reproducible similarity and retrieval baselines for **RAG/QA** workflows.

```
```
