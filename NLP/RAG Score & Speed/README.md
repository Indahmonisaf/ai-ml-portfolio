
# RAG Score Evaluation & Speed Analysis

Notebook ini membandingkan **kualitas retrieval** dan **kecepatan** untuk beberapa backend retrieval dalam pipeline RAG. Juga dibandingkan **category-aware retrieval** vs **global retrieval** pada dataset Q&A.

---

## Table of Contents
- [Dataset](#dataset)
- [Retrieval Backends Evaluated](#retrieval-backends-evaluated)
- [Scoring & Metrics](#scoring--metrics)
- [Environment & Dependencies](#environment--dependencies)
- [How to Run](#how-to-run)
- [Suggested Results Table (fill after running)](#suggested-results-table-fill-after-running)
- [Notes & Design Choices](#notes--design-choices)
- [Roadmap](#roadmap)
- [Author](#author)

---

## Dataset

**Format:** JSONL (satu record per Q&A):

```json
{"question": "...", "answer": "...", "category": "..."}
````

**Contoh nama file:** `output_with_categories_1107.jsonl`.

---

## Retrieval Backends Evaluated

* **SBERT** (`all-MiniLM-L6-v2`) + **ScaNN**
* **T5-base encoder** embeddings + **FAISS**
  *(encoder hidden-state mean pooling → vektor)*
* **TF-IDF** + **Annoy** *(angular ≈ cosine)*
* **Universal Sentence Encoder** (TF-Hub) + **HNSW** (*hnswlib*)
* **Word2Vec implementation**

> Setiap backend dievaluasi **dengan category** dan **tanpa category** untuk mengukur efek filtering/boosting berdasarkan kategori.

---

## Scoring & Metrics

* **Similarity Score (0–5):** Untuk Top-5 neighbors tiap query, prompt Vertex AI (Gemini 1.5 Flash) memberi skor **0–5** memakai rubric terstruktur.
* **Speed:** Rata-rata waktu retrieval per query (**detik**).
* **Query Set:** Sampel acak **20** query dari dataset.

> Notebook mencetak **Average Similarity Score** dan **Average Retrieval Time** per model/setting.
> *IR metrics klasik (Precision@K, Recall@K, MRR, nDCG) belum dihitung dan ada di roadmap.*

---

## Environment & Dependencies

Install library yang dipakai:

```bash
pip install \
  sentence-transformers scann faiss-cpu transformers torch \
  scikit-learn annoy jsonlines numpy \
  tensorflow tensorflow-hub hnswlib \
  google-cloud-aiplatform vertexai
```

Jika punya GPU yang kompatibel dan ingin FAISS-GPU (opsional):

```bash
pip install faiss-gpu
```

**Google Cloud / Vertex AI (untuk similarity scoring):**

1. Enable **Vertex AI API** di project GCP.
2. Set project & region default, contoh:

```python
import vertexai
vertexai.init(project="YOUR_GCP_PROJECT_ID", location="us-central1")
```

Notebook menggunakan **Gemini 1.5 Flash** untuk menghasilkan JSON seperti:

```json
{"similarity": 0-5, "comment": "penjelasan singkat"}
```

---

## How to Run

1. Letakkan dataset di root project (contoh: `output_with_categories_1107.jsonl`).
2. Buka notebook **`RAG Score Evaluation and Speed Analysis.ipynb`**.
3. Jalankan berurutan:

   * **Load data → Build vectors → Create index → Retrieve Top-5**
   * **Compute average similarity score (Vertex AI)**
   * **Measure average retrieval time**
   * **Ulangi dengan dan tanpa logika category (filter/boost)**
4. *(Opsional)* Simpan hasil ke **CSV** dan buat visualisasi.

---

## Suggested Results Table (fill after running)

| Model                      | Index | Category Mode  | Avg Similarity (0–5) | Avg Retrieval Time (s) |
| -------------------------- | ----- | -------------- | -------------------- | ---------------------- |
| SBERT (all-MiniLM-L6-v2)   | ScaNN | With / Without | …                    | …                      |
| T5-base (encoder mean)     | FAISS | With / Without | …                    | …                      |
| TF-IDF                     | Annoy | With / Without | …                    | …                      |
| Universal Sentence Encoder | HNSW  | With / Without | …                    | …                      |
| Word2Vec                   | —     | With / Without | …                    | …                      |

---

## Notes & Design Choices

* **Category-aware retrieval** mempersempit atau me-re-rank berdasarkan kategori agar neighborhood Top-K lebih relevan.
* **LLM-based similarity scoring** memberi ukuran yang lebih selaras dengan persepsi manusia (0–5) untuk perbandingan kualitatif lintas model; dapat dilengkapi IR metrics.
* **Speed** diukur sebagai rata-rata waktu retrieval per query—encoding dilakukan terlebih dulu per backend untuk mengisolasi kinerja index.

---

## Roadmap

* Tambah **Precision@K / Recall@K / MRR / nDCG**
* Persist semua run ke **CSV** dan tambah **matplotlib** plots
* Benchmark **latency vs recall** (ScaNN vs FAISS vs HNSW vs Annoy)
* Coba **hybrid search** (BM25/TF-IDF + embeddings) & **category-aware re-ranking**
* **Multilingual stress tests** (dataset punya `category`; JP tokenization via **Janome** bila perlu)

---

## Author

Built by **Indah Monisa Firdiantika (M.S.)** — evaluasi end-to-end kualitas dan kecepatan retrieval untuk sistem RAG, termasuk category-aware retrieval dan LLM-based similarity scoring.

```
```
