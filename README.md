# AI/ML Portfolio

A concise portfolio showcasing projects I built during my studies in machine learning, deep learning, and computer vision. Each project folder contains a short write-up (goal, data, method, metrics, how to run) and will include code/notebooks once uploaded.

---

## Table of Contents

### Computer Vision
- [Small Ship Detection in Infrared Images (IS-YOLO)](./Computer%20Vision/Small%20Ship%20Detection%20(IS-YOLO)/README.md) — YOLOv7-based detector tailored for small IR targets under cluttered backgrounds; improved backbone/neck for accuracy vs speed.
- [Box Recognition Method and Computing Device (EGCY-Net)](./Computer%20Vision/Box%20Recognition%20(EGCY-Net)/README.md) — Lightweight YOLO variant for stacked box type & count on embedded devices (Jetson Nano).
- [AI Vision for Overlapped Steel Sheet Detection](./Computer%20Vision/Overlapped%20Steel%20Sheet%20Detection/README.md) — Instance segmentation + overlap logic to catch double-feed sheets on production lines.
- [Forehead, Nose, and Cheek Detection (RGB & Infrared)](./Computer%20Vision/Face%20Areas%20(RGB+IR)/README.md) — Multimodal face-area detection using RGB+IR fusion.
- [AI Fire & Smoke Detection for Smart Manufacturing](./Computer%20Vision/Fire%20&%20Smoke%20Detection/README.md) — Early fire/smoke detection with temporal smoothing for low false alarms.

### Image Classification
- [Skin Cancer Classification (CNN)](./Image%20Classification/Skin%20Cancer%20Classification%20(CNN)/README.md) — MLP vs custom CNN vs VGG-16 TL on HAM10000; accuracy and latency compared.
- [Pneumonia Detection (CXR, CNN)](./Image%20Classification/Pneumonia%20Detection%20(CXR,%20CNN)/README.md) — ResNet-50/VGG-16 TL for pediatric chest X-rays; strong sensitivity/specificity.
- [Malaria Detection (Plasmodium Schizont & Gametocyte)](./Image%20Classification/Malaria%20Detection/README.md) — CNN classifier for infected cell patches.
- [Face Detection](./Image%20Classification/Face%20Detection/README.md) — Real-time face detection (MTCNN/RetinaFace) with FPS notes.

### Image-to-Image Translation
- [CycleGAN](./Image-to-Image/CycleGAN/README.md) — Unpaired image translation (A↔B) with identity loss notes.
- [Neural Style Transfer](./Image-to-Image/Neural%20Style%20Transfer/README.md) — Classic NST / perceptual loss; before/after gallery.

### Natural Language Processing
- [RAG Evaluation – Category-Aware Question Retrieval](./NLP/RAG%20Evaluation/README.md) — Compare global vs category-aware retrieval on Q&A dataset.
- [RAG Score Evaluation & Speed Analysis](./NLP/RAG%20Score%20&%20Speed/README.md) — Quality (0–5 rubric) vs latency across backends.
- [Similarity Assessment Model](./NLP/RAG%20Similarity%20Assessment/README.md) — Embeddings + ANN (FAISS/ScaNN/Annoy/HNSW) with Top-K scoring.
- [Module to Create Fine-Tuning](./NLP/Module%20to%20Create%20Fine-Tuning/README.md) — Modular LoRA/QLoRA fine-tuning workflow (Transformers + PEFT).
- [CRUD Operation – Minimal API & Data Layer](./Other/CRUD%20Operation%20–%20Minimal%20API%20&%20Data%20Layer/README.md) — Typed CRUD mini-stack for demos/tools.

### Machine Learning
- [Visualizing High-Dimensional Data with t-SNE](./Machine%20Learning/t-SNE%20Visualization/README.md) — Feature clustering visualization (t-SNE/UMAP).
- [Skin Cancer with Classical ML](./Machine%20Learning/Skin%20Cancer%20(Classical%20ML)/README.md) — SVM/RF baselines vs CNN.

### Other
- [Generating Synonym Words with Python](./Other/Generating%20Synonyms/README.md) — POS-aware synonym script for words/phrases.

---

## Tech Highlights
- **CV:** YOLOv7 variants, Mask R-CNN, deployment on Jetson Nano.
- **NLP:** RAG retrieval baselines, SBERT/ScaNN/FAISS, category-aware re-ranking.
- **ML Ops (lightweight):** ONNX export, timing/FPS notes, clean notebooks with “How to run”.

> **Author:** Indah Monisa Firdiantika 

