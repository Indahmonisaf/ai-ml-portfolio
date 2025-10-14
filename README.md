Here’s a polished `README.md` you can drop into the root of your repo. It keeps your link structure, adds clear summaries, and includes a consistent template for each project (overview, data, method, results, how to run, and media slots for images/videos).

---

# AI/ML Portfolio

A concise portfolio showcasing projects I built during my studies in machine learning, deep learning, and computer vision. Each project folder contains a short write-up (goal, data, method, metrics, how to run) and will include code/notebooks once uploaded.

> **Author:** Indah Monisa Firdiantika

---

## Table of Contents

### Computer Vision

* [Small Ship Detection in Infrared Images (IS-YOLO)](./Computer%20Vision/Small%20Ship%20Detection%20%28IS-YOLO%29/README.md) — YOLOv7-based detector tailored for tiny IR targets in clutter; improved backbone/neck for accuracy vs. speed.
* [Box Recognition Method and Computing Device (EGCY-Net)](./Computer%20Vision/Box%20Recognition%20%28EGCY-Net%29/README.md) — Lightweight YOLO variant for stacked-box type & count on edge/embedded devices.
* [AI Vision for Overlapped Steel Sheet Detection](./Computer%20Vision/Overlapped%20Steel%20Sheet%20Detection/README.md) — Instance segmentation + overlap logic to catch double-feed sheets on lines.
* [Forehead, Nose, and Cheek Detection (RGB & Infrared)](./Computer%20Vision/Face%20Areas%20%28RGB+IR%29/README.md) — Multimodal face-area detection using RGB+IR fusion.
* [AI Fire & Smoke Detection for Smart Manufacturing](./Computer%20Vision/Fire%20&%20Smoke%20Detection/README.md) — Early fire/smoke detection with temporal smoothing for low false alarms.

### Image Classification

* [Skin Cancer Classification (CNN)](./Image%20Classification/Skin%20Cancer%20Classification%20%28CNN%29/README.md) — MLP vs. custom CNN vs. VGG-16 transfer learning on HAM10000; accuracy/latency compared.
* [Pneumonia Detection](./Image%20Classification/Pneumonia%20Detection/README.md) — ResNet-50/VGG-16 TL for pediatric chest X-rays; sensitivity/specificity focus.
* [Malaria Detection (Plasmodium Schizont & Gametocyte)](./Image%20Classification/Malaria%20Detection/README.md) — CNN classifier for infected cell patches.
* [Face Detection](./Image%20Classification/Face%20Detection/README.md) — Real-time face detection (MTCNN/RetinaFace) with FPS notes.

### Image-to-Image Translation

* [CycleGAN](./Image-to-Image/CycleGAN/README.md) — Unpaired image translation (A↔B) with identity loss notes.
* [Neural Style Transfer](./Image-to-Image/Neural%20Style%20Transfer/README.md) — Classic NST / perceptual loss; before/after gallery.

### Natural Language Processing

* [RAG Evaluation – Category-Aware Question Retrieval](./NLP/RAG%20Evaluation/README.md) — Compare global vs. category-aware retrieval on a QA dataset.
* [RAG Score Evaluation & Speed Analysis](./NLP/RAG%20Score%20&%20Speed/README.md) — Quality rubric (0–5) vs. latency across vector DB/backends.
* [Similarity Assessment Model](./NLP/RAG%20Similarity%20Assessment/README.md) — Embeddings + ANN (FAISS/ScaNN/Annoy/HNSW) with Top-K scoring.
* [Module to Create Fine-Tuning](./NLP/Module%20to%20Create%20Fine-Tuning/README.md) — Modular LoRA/QLoRA pipelines (Transformers + PEFT).
* [CRUD Operation – Minimal API & Data Layer](./NLP/CRUD%20Operation%20–%20Minimal%20API%20&%20Data%20Layer/README.md) — Typed CRUD mini-stack for demos/tools.

### Machine Learning

* [t-SNE / UMAP Visualization](./Machine%20Learning/t-SNE%20Visualization/README.md) — Visualizing high-dimensional features.
* [Skin Cancer with Classical ML](./Machine%20Learning/Skin%20Cancer%20%28Classical%20ML%29/README.md) — SVM/RF baselines vs. CNN.

### Other

* [Generating Synonym Words with Python](./Other/Generating%20Synonyms/README.md) — POS-aware synonym script for words/phrases.

---

## Project Summaries

### 1) Small Ship Detection in Infrared Images (IS-YOLO)

**Folder:** `Computer Vision/Small Ship Detection (IS-YOLO)/`
**Overview:** Detector optimized for very small infrared ship targets under heavy background clutter (waves, noise, low contrast).
**Data:** Infrared maritime imagery; tiny objects with high class imbalance.
**Method:** YOLOv7 base with lightweight backbone tweaks, improved neck (feature pyramid), anchor/stride tuning for tiny targets, label-assignment/augmentation tailored to IR.
**Results:** Higher mAP for small objects at comparable FPS to baseline YOLOv7.
**How to Run:** See the [project README](./Computer%20Vision/Small%20Ship%20Detection%20%28IS-YOLO%29/README.md).
**Media:** *Add sample **images** and **videos** here once available.*

---

### 2) Box Recognition Method and Computing Device (EGCY-Net)

**Folder:** `Computer Vision/Box Recognition (EGCY-Net)/`
**Overview:** Real-time recognition of stacked boxes (type/count) for logistics/QC on edge devices.
**Data:** Factory/cargo images with occlusions and varied lighting.
**Method:** YOLO-style lightweight backbone + efficient heads; quantization and TensorRT/ONNX-runtime for Jetson-class deployment.
**Results:** Near real-time FPS on Jetson Nano with strong counting accuracy.
**How to Run:** See the [project README](./Computer%20Vision/Box%20Recognition%20%28EGCY-Net%29/README.md).
**Media:** *Add **images/videos** of inference in production.*

---

### 3) AI Vision for Overlapped Steel Sheet Detection

**Folder:** `Computer Vision/Overlapped Steel Sheet Detection/`
**Overview:** Prevent double-feed on lines by detecting overlapped metal sheets.
**Data:** Industrial line imagery; reflective surfaces, motion blur.
**Method:** Instance segmentation (Mask R-CNN/YOLO-seg) + post-processing overlap logic; temporal smoothing.
**Results:** High recall on overlaps with low false positives after temporal filtering.
**How to Run:** See the [project README](./Computer%20Vision/Overlapped%20Steel%20Sheet%20Detection/README.md).
**Media:** *Add **before/after** frames and a short **demo video**.*

---

### 4) Forehead, Nose, and Cheek Detection (RGB & Infrared)

**Folder:** `Computer Vision/Face Areas (RGB+IR)/`
**Overview:** Multimodal detection of facial areas (forehead, nose, cheeks) using fused RGB+IR cues.
**Data:** Paired RGB/IR images; varied skin tones and lighting.
**Method:** Dual-stream backbone, feature fusion, small anchor sizes; calibration between modalities.
**Results:** More robust localization vs. RGB-only models in low light.
**How to Run:** See the [project README](./Computer%20Vision/Face%20Areas%20%28RGB+IR%29/README.md).
**Media:** *Insert **RGB/IR visualizations** and overlays.*

---

### 5) AI Fire & Smoke Detection for Smart Manufacturing

**Folder:** `Computer Vision/Fire & Smoke Detection/`
**Overview:** Early detection of fire/smoke for safety monitoring in factories.
**Data:** Video streams with environmental noise (steam, dust).
**Method:** Frame-level detector + motion/temporal filters; optional optical-flow cue to cut false alarms.
**Results:** Reduced false positives while keeping early-warning sensitivity.
**How to Run:** See the [project README](./Computer%20Vision/Fire%20&%20Smoke%20Detection/README.md).
**Media:** *Add **alert clips** and **ROC/PR** curves.*

---

### 6) Skin Cancer Classification (CNN)

**Folder:** `Image Classification/Skin Cancer Classification (CNN)/`
**Overview:** Compare MLP, custom CNN, and VGG-16 TL on HAM10000 skin lesion dataset.
**Data:** HAM10000 dermoscopy images; class imbalance handled with augmentation/weighted loss.
**Method:** Transfer learning (VGG-16), focal loss/weighted CE, early stopping; latency measured.
**Results:** Best accuracy with VGG-16 TL; CNN competitive with lower latency.
**How to Run:** See the [project README](./Image%20Classification/Skin%20Cancer%20Classification%20%28CNN%29/README.md).
**Media:** *Confusion matrix and sample predictions.*

---

### 7) Pneumonia Detection

**Folder:** `Image Classification/Pneumonia Detection/`
**Overview:** TL models (ResNet-50/VGG-16) for pediatric chest X-rays.
**Data:** Pediatric CXR dataset (normal vs. pneumonia).
**Method:** TL with careful augmentation; emphasis on recall (sensitivity).
**Results:** High sensitivity/specificity with good calibration.
**How to Run:** See the [project README](./Image%20Classification/Pneumonia%20Detection/README.md).
**Media:** *Grad-CAM heatmaps and ROC curves.*

---

### 8) Malaria Detection (Plasmodium Schizont & Gametocyte)

**Folder:** `Image Classification/Malaria Detection/`
**Overview:** Classify infected cell patches by parasite stage.
**Data:** Microscopy patches (balanced after augmentation).
**Method:** Custom CNN with small receptive fields; patch normalization.
**Results:** Strong per-class F1; portable inference script.
**How to Run:** See the [project README](./Image%20Classification/Malaria%20Detection/README.md).
**Media:** *Patch gallery and confusion matrix.*

---

### 9) Face Detection

**Folder:** `Image Classification/Face Detection/`
**Overview:** Real-time face detection with MTCNN/RetinaFace and FPS benchmarks.
**Data:** Webcam/video frames; varied poses and light.
**Method:** Pretrained detectors + ONNX export; batch vs. stream profiles.
**Results:** Stable FPS with acceptable miss rate on edge GPUs/CPU.
**How to Run:** See the [project README](./Image%20Classification/Face%20Detection/README.md).
**Media:** *Short demo clips and latency table.*

---

### 10) CycleGAN

**Folder:** `Image-to-Image/CycleGAN/`
**Overview:** Unpaired image-to-image translation (domain A↔B).
**Data:** Two unpaired domains (e.g., summer↔winter, horses↔zebras).
**Method:** Cycle consistency + identity loss; patchGAN discriminator.
**Results:** Visually plausible translations with minimal mode collapse.
**How to Run:** See the [project README](./Image-to-Image/CycleGAN/README.md).
**Media:** *Before/after grids and FID (optional).*

---

### 11) Neural Style Transfer

**Folder:** `Image-to-Image/Neural Style Transfer/`
**Overview:** Classic NST using perceptual VGG features.
**Data:** Content + style images; multiple style-weights.
**Method:** Optimize content image; total variation regularization.
**Results:** Clean stylizations; timing across image sizes.
**How to Run:** See the [project README](./Image-to-Image/Neural%20Style%20Transfer/README.md).
**Media:** *Style gallery and parameter sweeps.*

---

### 12) RAG Evaluation – Category-Aware Question Retrieval

**Folder:** `NLP/RAG Evaluation/`
**Overview:** Does category-aware retrieval beat global retrieval on QA?
**Data:** Q&A pairs with category labels.
**Method:** SBERT embeddings; filtering/re-ranking by category.
**Results:** Higher hit@k and MRR in category-aware setting.
**How to Run:** See the [project README](./NLP/RAG%20Evaluation/README.md).
**Media:** *Precision/recall plots and latency bars.*

---

### 13) RAG Score Evaluation & Speed Analysis

**Folder:** `NLP/RAG Score & Speed/`
**Overview:** Trade-offs between response quality and latency across stacks.
**Data:** Fixed prompt set, human rubric (0–5).
**Method:** Backends (FAISS/ScaNN/HNSW) and chunk sizes; measure end-to-end time.
**Results:** Clear Pareto frontier for speed vs. quality.
**How to Run:** See the [project README](./NLP/RAG%20Score%20&%20Speed/README.md).
**Media:** *Spider/radar charts.*

---

### 14) Similarity Assessment Model

**Folder:** `NLP/RAG Similarity Assessment/`
**Overview:** Embedding search with ANN for top-K similarity scoring.
**Data:** Text pairs/paragraphs.
**Method:** SBERT embeddings; FAISS/Annoy/HNSW comparisons.
**Results:** Consistent top-K stability at low query latency.
**How to Run:** See the [project README](./NLP/RAG%20Similarity%20Assessment/README.md).
**Media:** *t-SNE/UMAP plots.*

---

### 15) Module to Create Fine-Tuning

**Folder:** `NLP/Module to Create Fine-Tuning/`
**Overview:** Reusable LoRA/QLoRA training module with PEFT.
**Data:** JSONL/text datasets; config-driven.
**Method:** HF Transformers + PEFT; mixed-precision; checkpoints.
**Results:** Reproducible deltas; quick ablations.
**How to Run:** See the [project README](./NLP/Module%20to%20Create%20Fine-Tuning/README.md).
**Media:** *Loss curves and sample outputs.*

---

### 16) CRUD Operation – Minimal API & Data Layer

**Folder:** `NLP/CRUD Operation – Minimal API & Data Layer/`
**Overview:** Minimal, typed CRUD API + data access layer for demos/tools.
**Stack:** FastAPI/Express (option), SQL/SQLite, pydantic/TypeScript types.
**How to Run:** See the [project README](./NLP/CRUD%20Operation%20–%20Minimal%20API%20&%20Data%20Layer/README.md).
**Media:** *ERD diagram and request/response examples.*

---

### 17) t-SNE / UMAP Visualization

**Folder:** `Machine Learning/t-SNE Visualization/`
**Overview:** Visual exploration of high-dimensional features.
**Data:** Extracted embeddings from CV/NLP tasks.
**Method:** t-SNE/UMAP with perplexity/neighbors sweeps; cluster labels.
**Results:** Clear cluster separation; aids error analysis.
**How to Run:** See the [project README](./Machine%20Learning/t-SNE%20Visualization/README.md).
**Media:** *2D plots and clustering overlays.*

---

### 18) Skin Cancer with Classical ML

**Folder:** `Machine Learning/Skin Cancer (Classical ML)/`
**Overview:** SVM/Random Forest baselines vs. CNN for skin lesion classification.
**Data:** Tabularized features (color/texture) from lesions.
**Method:** HOG/LBP features; SVM/RF with CV; compare to CNN.
**Results:** CNN wins overall; classical ML competitive with few features.
**How to Run:** See the [project README](./Machine%20Learning/Skin%20Cancer%20%28Classical%20ML%29/README.md).
**Media:** *ROC/PR and feature importance.*

---

### 19) Generating Synonym Words with Python

**Folder:** `Other/Generating Synonyms/`
**Overview:** POS-aware synonym generator for words/phrases.
**Data:** Word lists; optional corpora for frequency.
**Method:** POS tagging → synonym lookup → filters (frequency/stopwords).
**Results:** Cleaner substitutions for paraphrasing/humanizing tasks.
**How to Run:** See the [project README](./Other/Generating%20Synonyms/README.md).
**Media:** *CLI GIF and sample outputs.*

---

## Tech Highlights

* **Computer Vision:** YOLOv7 variants, instance segmentation, RGB+IR fusion, ONNX/TensorRT for edge deployment.
* **NLP / RAG:** SBERT, FAISS/ScaNN/HNSW, category-aware retrieval/reranking, human rubric scoring.
* **ML Ops (lightweight):** ONNX export, timing/FPS notes, reproducible notebooks with “How to run”.

---

### Notes

* Each **Table of Contents** item links to the project’s own `README.md` in its folder.
* Add your **images/videos** inside each project folder and reference them in that project’s README.
* As code/notebooks are uploaded, update the **How to Run** sections with exact commands.
