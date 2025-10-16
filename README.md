

# AI/ML Portfolio

A concise portfolio showcasing projects I built during my studies in machine learning, deep learning, and computer vision. Each project folder includes a short write-up (goal, data, method, metrics, how to run). 

> **Author:** Indah Monisa Firdiantika

---

## Table of Contents

* **Computer Vision**

  * [Box Recognition Method and Computing Device (EGCY-Net)](./Computer%20Vision/Box%20Recognition%20%28EGCY-Net%29/README.md)
  * [Small Ship Detection in Infrared Images (IS-YOLO)](./Computer%20Vision/Small%20Ship%20Detection%20%28IS-YOLO%29/README.md)
  * [Forehead, Nose, and Cheek Detection (RGB & Infrared)](./Computer%20Vision/Face%20Areas%20%28RGB+IR%29/README.md)
  * [AI Vision for Overlapped Steel Sheet Detection](./Computer%20Vision/Overlapped%20Steel%20Sheet%20Detection/README.md)

* **Image Classification**

  * [Skin Cancer Classification (CNN)](./Image%20Classification/Skin%20Cancer%20Classification%20%28CNN%29/README.md)
  * [Pneumonia Detection](./Image%20Classification/Pneumonia%20Detection/README.md)

* **Image-to-Image Translation**

  * [Neural Style Transfer](./Image-to-Image/Neural%20Style%20Transfer/README.md)

* **Natural Language Processing**

  * [RAG Evaluation – Category-Aware Question Retrieval](./NLP/RAG%20Evaluation/README.md)
  * [RAG Score Evaluation & Speed Analysis](./NLP/RAG%20Score%20&%20Speed/README.md)
  * [Similarity Assessment Model](./NLP/RAG%20Similarity%20Assessment/README.md)
  * [Module to Create Fine-Tuning](./NLP/Module%20to%20Create%20Fine-Tuning/README.md)
  * [CRUD Operation – Minimal API & Data Layer](./NLP/CRUD%20Operation%20–%20Minimal%20API%20&%20Data%20Layer/README.md)


* **Web Development**

  * [Generating Synonym Words with Python](./Web%20Development/Generating%20Synonym%20Words%20with%20Python/README.md)
  * [AI Writing Assistant Website](./Web%20Development/AI%20Writing%20Assistant%20Website/README.md)

---

## Project Summaries

### Computer Vision

* **Small Ship Detection in Infrared Images (IS-YOLO)** — YOLOv7-based detector tailored for tiny IR targets in clutter; tuned backbone/neck, anchors, and augmentation for small objects. <br>
 **Paper:** [IS-YOLO: A YOLOv7-based Detection Method for Small Ship Detection in Infrared Images With Heterogeneous Backgrounds — Springer](https://link.springer.com/article/10.1007/s12555-024-0044-8)
* **Box Recognition (EGCY-Net)** — Lightweight YOLO variant to detect/ count stacked boxes on edge devices (Jetson); quantization + TensorRT/ONNX for real-time. <br>
  **Paper:** [EGCY-Net: An ELAN and GhostConv-Based YOLO Network for Stacked Packages in Logistic Systems - MDPI](https://www.mdpi.com/2076-3417/14/7/2763)
* **Overlapped Steel Sheet Detection** — Instance segmentation + overlap logic + temporal smoothing to prevent double-feed on lines.
* **Face Areas (RGB+IR)** — Multimodal detection of forehead/nose/cheeks with RGB–IR fusion; robust in low light.

### Image Classification

* **Skin Cancer (CNN)** — Compare MLP, custom CNN, and VGG-16 TL on HAM10000; best accuracy with TL, latency compared. <br>
  **Paper:** [Performance of Multi Layer Perceptron and Deep Neural Networks in Skin Cancer Classification - 2021 IEEE 3rd Global Conference on Life Sciences and Technologies (LifeTech) ](https://ieeexplore.ieee.org/document/9391876)
* **Pneumonia Detection** — ResNet-50/VGG-16 TL on pediatric CXR; sensitivity/specificity focus with careful augmentation. <br>
  **Paper:** [Pneumonia detection in chest X-ray images using convolutional neural network](https://pubs.aip.org/aip/acp/article-abstract/2499/1/020001/2827211/Pneumonia-detection-in-chest-X-ray-images-using?redirectedFrom=fulltext)
* **Malaria Detection** — CNN for parasite stages (schizont/gametocyte) on microscopy patches; balanced with augmentation. <br>
  **Paper:** [Classification of Plasmodium Skizon and Gametocytes Malaria Images Using Deep Learning](https://ieeexplore.ieee.org/document/9649676)

### Image-to-Image Translation

* **Neural Style Transfer** — VGG perceptual loss, TV regularization; before/after gallery.

### Natural Language Processing

* **RAG Evaluation – Category-Aware Retrieval** — SBERT embeddings + category filter/rerank vs global retrieval; higher hit@k/MRR.
* **RAG Score & Speed** — Quality rubric (0–5) vs latency across vector DB/backends and chunking configs; Pareto trade-off.
* **Similarity Assessment Model** — ANN search (FAISS/ScaNN/Annoy/HNSW) on embeddings; stable top-K at low latency.
* **Module to Create Fine-Tuning** — Configurable LoRA/QLoRA pipelines (Transformers + PEFT); mixed precision; checkpoints.
* **CRUD Operation – Minimal API & Data Layer** — Typed FastAPI/Express baseline with SQL/SQLite; clean DAL for demos.

### Other

* **Generating Synonym Words with Python** — POS-aware synonym script for words/phrases; filters by frequency/stopwords for cleaner substitutions.

---

## Tech Highlights

* **CV:** Python YOLOv7 variants, instance segmentation, RGB+IR fusion, ONNX/TensorRT deployment.
* **NLP / RAG:** SBERT embeddings, FAISS/ScaNN/HNSW, category-aware reranking, human quality rubric.
* **Lightweight MLOps:** ONNX export, timing/FPS notes”.

