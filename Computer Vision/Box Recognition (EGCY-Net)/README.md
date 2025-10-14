### EGCY-Net: ELAN & GhostConv-Based YOLO for Stacked Packages in Logistic Systems

> End-to-end vision system that **classifies box availability**, **detects box types (`04`, `13`, `A`, `Y`)**, and **counts quantities** for AGV-driven logistics. Implemented on **Jetson Nano + Arducam IMX477** and benchmarked against SOTA detectors. :contentReference[oaicite:0]{index=0}


## TL;DR / Highlights
- **Two-stage pipeline**: (1) **Box availability** classifier (MobileNet) on cropped RoI; (2) **EGCY-Net** detector (YOLOv7 base with **CGStack** & **EGCNet**) to identify box **type** and **count**. :contentReference[oaicite:1]{index=1}  
- **Dataset**: Factory images/videos → curated & labeled; **4 box types (04, 13, A, Y)**; **1258 images** for detection; availability set: **1200 “box” + 1200 “no box”** frames. :contentReference[oaicite:2]{index=2}  
- **Results (20-epoch comparison)**: EGCY-Net achieves **best mAP@0.5:0.95** on all/most classes with **fewest params (~30M)**, **smallest ONNX (111MB)**, and **fastest inference (5.96s)** among tested models. :contentReference[oaicite:3]{index=3}  
- **Embedded deployment**: Converted **PyTorch → ONNX**, optimized for **Jetson Nano**. :contentReference[oaicite:4]{index=4}

---

## Demo Media (drop your evidence here)

### 📸 Photos (qualitative results)
Place images in `./figures/results/` and reference them below.

<p align="center">
  <img src="./figures/results/sample_01.jpg" alt="Result 1" width="45%"/>
  <img src="./figures/results/sample_02.jpg" alt="Result 2" width="45%"/>
</p>

### 🎥 Videos (detections / AGV workflow)
Put videos in `./videos/` (MP4 recommended), then link:

- [Detection Demo (Smartphone)](./videos/detect_smartphone.mp4)  
- [Detection Demo (Arducam IMX477)](./videos/detect_arducam.mp4)  
- [End-to-End AGV Workflow](./videos/agv_workflow.mp4)

> Tip: GitHub can preview small MP4s inline. For large files, consider **Git LFS** or link to Releases.

---

## Abstract
Manual recognition/counting of stacked packages slows logistics. We present **EGCY-Net**, a one-stage YOLOv7-based detector enhanced with a **Conv-GhostConv Stack (CGStack)** and an **ELAN-GhostConv module (EGCNet)** to better capture hierarchical features with fewer parameters. Together with a lightweight **MobileNet** classifier for **box/no-box** availability, the system achieves high precision/recall and competitive mAP while remaining deployable on embedded hardware. :contentReference[oaicite:5]{index=5}

---

## Dataset

### 1) Box Availability (classification)
- **Classes**: `box`, `no box`  
- **Source**: Factory video → **RoI cropping** → frame extraction  
- **Size**: **1200** images per class (JPG)  
- **Usage**: Train MobileNet for availability in storage RoIs. :contentReference[oaicite:6]{index=6}

### 2) Box Type & Quantity (detection)
- **Classes**: `04`, `13`, `A`, `Y`  
- **Total images**: **1258** (JPG), collected from smartphone/Arducam; cropped to single stacks for speed. :contentReference[oaicite:7]{index=7}  
- **YOLO format** with txt labels: `<class cx cy w h>` (normalized)  
- **Suggested split (paper)**:
  - Train/Val/Test per class (example table in paper): `04 (420/80/12), 13 (420/80/34), A (420/80/40), Y (420/80/22)`. :contentReference[oaicite:8]{index=8}

> **Data availability**: Proprietary factory data; access on request with company permission. :contentReference[oaicite:9]{index=9}

---

## Method

### Pipeline
1) **Availability**: MobileNet on RoI → `box` / `no box`. :contentReference[oaicite:10]{index=10}  
2) **Type & Count**: **EGCY-Net** (YOLOv7 backbone) → per-stack detection; then **count boxes per class** with size/score thresholds (pseudo-code in paper). :contentReference[oaicite:11]{index=11}

### EGCY-Net Architecture (overview)
- **Backbone**: `Conv → EGCNet (ELAN + CGStack) → MP`  
- **Neck**: `Conv/Upsample/MP/Concat + EGCNet` with **multi-scale feature fusion**  
- **Head**: `IDetect` (three scales: S/M/L)  
- **CGStack**: `1×1 Conv → GhostConv → 1×1 Conv` (reduces params while preserving spatial features)  
- **EGCNet**: ELAN-style layer aggregation + two CGStacks to stabilize gradients and enrich features. :contentReference[oaicite:12]{index=12}

<p align="center">
  <img src="./figures/architecture.png" alt="EGCY-Net Architecture" width="85%"/>
</p>

---

## Implementation
- **Training env (paper)**: Linux, CUDA 11.8, cuDNN 8.1; RTX 4090 for training; **Jetson Nano** for deployment; **Arducam IMX477** camera. :contentReference[oaicite:13]{index=13}  
- **Formats**: Train in PyTorch → export **ONNX** for Jetson. :contentReference[oaicite:14]{index=14}
- **Availability model**: Input ~`128×128×3`, ~15 epochs, batch 8 (paper example). :contentReference[oaicite:15]{index=15}

---

## Results

### Comparison vs SOTA (20 epochs, paper setting)
**EGCY-Net** vs **YOLOv3 / YOLOv5l / YOLOR / YOLOv7 / YOLOv7x**:

- **All classes (mAP@0.5:0.95)**: **EGCY-Net 87.1%** (best)  
- **Per class (mAP@0.5:0.95)**: `04: 86.5%`, `13: 87.3%`, `A: 86.6%`, `Y: 88.6%`  
- **Params**: **~30M** (fewest) • **ONNX**: **111MB** (smallest) • **Inference time**: **5.96 s** (fastest of compared set)  
(Table and qualitative examples in the paper.) :contentReference[oaicite:16]{index=16}

### Visualization & Domain Robustness
t-SNE feature maps indicate **better class separation** and **cross-camera robustness** (smartphone ↔ Arducam) for EGCY-Net vs YOLOv7. :contentReference[oaicite:17]{index=17}

> For your repo, drop comparison screenshots under `./figures/results/` and short clips in `./videos/`.

---

## Publication
**EGCY-Net: An ELAN and GhostConv-Based YOLO Network for Stacked Packages in Logistic Systems**  
*Applied Sciences*, **14**(7):2763, **26 March 2024**.  
Authors: **Indah Monisa Firdiantika**, Seongryeong Lee, Chaitali Bhattacharyya, Yewon Jang, **Sungho Kim***.  
DOI: `10.3390/app14072763`. :contentReference[oaicite:18]{index=18}

- **Paper PDF**: put your copy at `./paper.pdf`

---

## Patent
**Artificial neural network for automated box recognition, box recognition method and computing device, and recording medium thereof**  
KIPO application no. **10-2024-0046794**, **filed 2024-04-05**.  
Applicant: **Yeungnam University Industry-Academic Cooperation Foundation**.  
Inventors: **Sungho Kim**, **Indah Monisa Firdiantika**, Seongryeong Lee, Chaitali Bhattacharyya, Yewon Jang. :contentReference[oaicite:19]{index=19}

- **Patent PDF**: put your copy at `./patent.pdf`

---

## Folder Structure
```

projects/egcy-net/
├─ README.md
├─ paper.pdf           # Applied Sciences paper
├─ patent.pdf          # KIPO application PDF
├─ figures/
│  ├─ cover.jpg
│  ├─ architecture.png
│  └─ results/
│     ├─ sample_01.jpg
│     └─ sample_02.jpg
└─ videos/
├─ detect_smartphone.mp4
├─ detect_arducam.mp4
└─ agv_workflow.mp4

````

---

## Citation
If you use this work, please cite:

```bibtex
@article{Firdiantika2024EGCYNet,
  title   = {EGCY-Net: An ELAN and GhostConv-Based YOLO Network for Stacked Packages in Logistic Systems},
  author  = {Firdiantika, Indah Monisa and Lee, Seongryeong and Bhattacharyya, Chaitali and Jang, Yewon and Kim, Sungho},
  journal = {Applied Sciences},
  year    = {2024},
  volume  = {14},
  number  = {7},
  pages   = {2763},
  doi     = {10.3390/app14072763}
}
````

---

## License & Data Use

* Paper is **CC BY 4.0**; credit the authors/publisher when reusing figures/tables.
* Factory dataset is **restricted**; contact authors/company for permissions. 

---

## Contact

* **Indah Monisa Firdiantika, M.S.** — add your email / LinkedIn
* Department of Electronic Engineering, Yeungnam University

```

---

