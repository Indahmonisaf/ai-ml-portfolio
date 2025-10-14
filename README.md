

# AI/ML Portfolio

A concise portfolio showcasing projects I built during my studies in machine learning, deep learning, and computer vision. Each project folder includes a short write-up (goal, data, method, metrics, how to run). Code/notebooks will be added as I upload them.

> **Author:** Indah Monisa Firdiantika

---

## Table of Contents

* **Computer Vision**

  * [Small Ship Detection in Infrared Images (IS-YOLO)](./Computer%20Vision/Small%20Ship%20Detection%20%28IS-YOLO%29/README.md)
  * [Box Recognition Method and Computing Device (EGCY-Net)](./Computer%20Vision/Box%20Recognition%20%28EGCY-Net%29/README.md)
  * [AI Vision for Overlapped Steel Sheet Detection](./Computer%20Vision/Overlapped%20Steel%20Sheet%20Detection/README.md)
  * [Forehead, Nose, and Cheek Detection (RGB & Infrared)](./Computer%20Vision/Face%20Areas%20%28RGB+IR%29/README.md)
  * [AI Fire & Smoke Detection for Smart Manufacturing](./Computer%20Vision/Fire%20&%20Smoke%20Detection/README.md)

* **Image Classification**

  * [Skin Cancer Classification (CNN)](./Image%20Classification/Skin%20Cancer%20Classification%20%28CNN%29/README.md)
  * [Pneumonia Detection](./Image%20Classification/Pneumonia%20Detection/README.md)
  * [Malaria Detection (Plasmodium Schizont & Gametocyte)](./Image%20Classification/Malaria%20Detection/README.md)
  * [Face Detection](./Image%20Classification/Face%20Detection/README.md)

* **Image-to-Image Translation**

  * [CycleGAN](./Image-to-Image/CycleGAN/README.md)
  * [Neural Style Transfer](./Image-to-Image/Neural%20Style%20Transfer/README.md)

* **Natural Language Processing**

  * [RAG Evaluation – Category-Aware Question Retrieval](./NLP/RAG%20Evaluation/README.md)
  * [RAG Score Evaluation & Speed Analysis](./NLP/RAG%20Score%20&%20Speed/README.md)
  * [Similarity Assessment Model](./NLP/RAG%20Similarity%20Assessment/README.md)
  * [Module to Create Fine-Tuning](./NLP/Module%20to%20Create%20Fine-Tuning/README.md)
  * [CRUD Operation – Minimal API & Data Layer](./NLP/CRUD%20Operation%20–%20Minimal%20API%20&%20Data%20Layer/README.md)

* **Machine Learning**

  * [t-SNE / UMAP Visualization](./Machine%20Learning/t-SNE%20Visualization/README.md)
  * [Skin Cancer with Classical ML](./Machine%20Learning/Skin%20Cancer%20%28Classical%20ML%29/README.md)

* **Other**

  * [Generating Synonym Words with Python](./Other/Generating%20Synonyms/README.md)

---

## Project Summaries

### Computer Vision

* **Small Ship Detection in Infrared Images (IS-YOLO)** — YOLOv7-based detector tailored for tiny IR targets in clutter; tuned backbone/neck, anchors, and augmentation for small objects. <br>
 **Paper:** [IS-YOLO: A YOLOv7-based Detection Method for Small Ship Detection in Infrared Images With Heterogeneous Backgrounds — Springer](https://link.springer.com/article/10.1007/s12555-024-0044-8)
* **Box Recognition (EGCY-Net)** — Lightweight YOLO variant to detect/ count stacked boxes on edge devices (Jetson); quantization + TensorRT/ONNX for real-time.
* **Overlapped Steel Sheet Detection** — Instance segmentation + overlap logic + temporal smoothing to prevent double-feed on lines.
* **Face Areas (RGB+IR)** — Multimodal detection of forehead/nose/cheeks with RGB–IR fusion; robust in low light.
* **Fire & Smoke Detection** — Early event detection with motion/temporal filters to reduce false alarms in factories.

### Image Classification

* **Skin Cancer (CNN)** — Compare MLP, custom CNN, and VGG-16 TL on HAM10000; best accuracy with TL, latency compared.
* **Pneumonia Detection** — ResNet-50/VGG-16 TL on pediatric CXR; sensitivity/specificity focus with careful augmentation.
* **Malaria Detection** — CNN for parasite stages (schizont/gametocyte) on microscopy patches; balanced with augmentation.
* **Face Detection** — Real-time MTCNN/RetinaFace with ONNX export; FPS benchmarks on CPU/edge GPU.

### Image-to-Image Translation

* **CycleGAN** — Unpaired A↔B translation with cycle consistency & identity loss; notes on stability.
* **Neural Style Transfer** — VGG perceptual loss, TV regularization; before/after gallery.

### Natural Language Processing

* **RAG Evaluation – Category-Aware Retrieval** — SBERT embeddings + category filter/rerank vs global retrieval; higher hit@k/MRR.
* **RAG Score & Speed** — Quality rubric (0–5) vs latency across vector DB/backends and chunking configs; Pareto trade-off.
* **Similarity Assessment Model** — ANN search (FAISS/ScaNN/Annoy/HNSW) on embeddings; stable top-K at low latency.
* **Module to Create Fine-Tuning** — Configurable LoRA/QLoRA pipelines (Transformers + PEFT); mixed precision; checkpoints.
* **CRUD Operation – Minimal API & Data Layer** — Typed FastAPI/Express baseline with SQL/SQLite; clean DAL for demos.

### Machine Learning

* **t-SNE / UMAP Visualization** — High-dimensional feature exploration; perplexity/neighbors sweeps and labeled clusters.
* **Skin Cancer (Classical ML)** — HOG/LBP + SVM/RF baselines vs CNN; PR/ROC and feature importance.

### Other

* **Generating Synonym Words with Python** — POS-aware synonym script for words/phrases; filters by frequency/stopwords for cleaner substitutions.

---

## Tech Highlights

* **CV:** YOLOv7 variants, instance segmentation, RGB+IR fusion, ONNX/TensorRT deployment.
* **NLP / RAG:** SBERT embeddings, FAISS/ScaNN/HNSW, category-aware reranking, human quality rubric.
* **Lightweight MLOps:** ONNX export, timing/FPS notes, reproducible notebooks with “How to run”.

> Tip: Add images/videos to each project folder and reference them inside that project’s `README.md` (e.g., `![demo](./media/demo.jpg)` or embedded MP4/GIF).
