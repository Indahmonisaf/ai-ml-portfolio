````markdown
# RAG Evaluation – Category-Aware Question Retrieval

**Notebook:** `RAG Evaluation.ipynb`  
This project evaluates **retrieval quality** for a Retrieval-Augmented Generation (RAG) system on a Q&A dataset. It compares **global retrieval** (no category) vs **category-aware retrieval** (filter/boost by category) and experiments with multiple **text representations** and **ANN indexes**.

---

## Table of Contents
- [Dataset](#dataset)
- [Approaches](#approaches)
  - [Representations](#representations)
  - [ANN Index](#ann-index)
  - [Pre-processing](#pre-processing)
- [Evaluation Protocol](#evaluation-protocol)
- [Example (qualitative)](#example-qualitative)
- [How to Run](#how-to-run)
- [Findings (current state)](#findings-current-state)
- [Roadmap](#roadmap)
- [Tech Stack](#tech-stack)
- [Author](#author)

---

## Dataset

**Format:** JSONL with fields:
```json
{"question": "...", "answer": "...", "category": "..."}
````

**Example filename:** `output_with_categories_1107.jsonl`.

---

## Approaches

### Representations

* **TF-IDF** + cosine similarity *(lexical baseline)*
* **Sentence embeddings** via `sentence-transformers` *(semantic)*
* **Word2Vec** *(distributional baseline)*

### ANN Index

* **ScaNN** *(Faiss is referenced in code for future swaps)*

### Pre-processing

* **Janome tokenizer** noted for **Japanese text** support *(multilingual-ready)*

---

## Evaluation Protocol

1. Hold out ~**20** questions as **queries**.
2. Retrieve **Top-5** most similar questions from the corpus.
3. Compare **Without Category** vs **With Category** to observe changes in neighborhood relevance.
4. The notebook currently **prints Top-K neighbors** for qualitative inspection.
5. **Planned metrics:** *Precision@K, Recall@K, MRR, nDCG*.

---

## Example (qualitative)

**Query:** “Is credit card payment available?”
**Top-5 (sample):**

1. “Do you accept credit card payment?”
2. “Do you accept credit card payments?”
3. “What payment methods are available?”
4. “What are the payment options?”

> *For illustration only; actual neighbors depend on the index/model and dataset.*

---

## How to Run

### 1) Install dependencies

```bash
pip install \
  sentence-transformers scann jsonlines numpy scikit-learn \
  faiss-cpu janome
```

> `faiss-cpu` and `janome` are optional depending on your needs.

### 2) Place the dataset

* Put `output_with_categories_1107.jsonl` at the **project root**.

### 3) Run the notebook

Open **`RAG Evaluation.ipynb`** and execute cells in order:

1. **Load data**
2. **Build representations** (TF-IDF / embeddings / Word2Vec)
3. **Index** (ScaNN / optional Faiss)
4. **Retrieve Top-K**
5. **Compare With vs Without Category**

### 4) (Optional) Add metrics & export

* Add cells for **P@K, R@K, MRR, nDCG**
* Export results to **CSV** for further analysis

---

## Findings (current state)

* The notebook demonstrates the **setup** and **comparison path** for **category-aware retrieval** vs **global retrieval**.
* It prints **Top-5 neighbors** for manual inspection; **quantitative metrics** are recommended as the next step.

---

## Roadmap

* Implement **Precision@K / Recall@K / MRR / nDCG**
* Add a **BM25 baseline** and **hybrid search** (BM25 + embeddings)
* Benchmark **Faiss vs ScaNN** *(latency, memory, recall)*
* Add **category-aware re-ranking** and ablation studies
* **Log & visualize** results (CSV + plots)

---

## Tech Stack

* **Python**, **Jupyter Notebook**
* `sentence-transformers`, **ScaNN** (ANN), `jsonlines`, `numpy`, `scikit-learn`, *(optional)* `faiss-cpu`, `janome`

---

## Author

Built by **Indah Monisa Firdiantika (M.S.)** — focused on practical RAG evaluation and **category-aware retrieval**.

```
```
