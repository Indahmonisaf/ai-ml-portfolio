
**“Small Ship Detection in Infrared Images (IS-YOLO)”**

> YOLOv7-based detector tailored for **small ship** targets in **infrared (IR)** imagery with heterogeneous sea backgrounds.

## TL;DR / Highlights
- **IS-YOLO** introduces lightweight yet effective backbone/neck upgrades (E-C2G, ResC3, **MPPELAN**) to improve **small-target** recognition in IR scenes with glare, clutter, waves, fog, islands, and shorelines. :contentReference[oaicite:0]{index=0}
- Trained and validated on a **FLIR T620**–captured small-ship IR dataset from Yeungnam University (**1,370 train / 120 val**), with additional tests on **SIRST-v2** and **IRSTD-1k** for generalization. :contentReference[oaicite:1]{index=1}
- Achieves **AP@0.5 = 88.9%** and **AP@0.5:0.95 = 38.3%**, outperforming strong baselines (YOLOv7/8/9, Faster R-CNN, RetinaNet, FCOS) while keeping parameters and model size competitive. :contentReference[oaicite:2]{index=2}

---
## Publication
**IS-YOLO: A YOLOv7-based Detection Method for Small Ship Detection in Infrared Images With Heterogeneous Backgrounds**  
*International Journal of Control, Automation, and Systems*, 22(11), 3295–3302, 2024.  
DOI: 10.1007/s12555-024-0044-8. :contentReference[oaicite:16]{index=16}

> **Paper PDF:** `./paper.pdf` (place your copy here)

---

## Patent
**Method for detecting small object in infrared image and system thereof**  
Korean Patent Application No. **10-2024-0040244**, filed **2024-03-25**; Applicant: **Yeungnam University Industry-Academic Cooperation Foundation**; Inventors: **Sungho Kim, Indah Monisa Firdiantika**. :contentReference[oaicite:17]{index=17}

> **Patent PDF:** `./patent.pdf` (place your copy here)

---

## Abstract
Ship detection in infrared imagery is challenging due to **low signal-to-clutter ratios**, indistinct contours, small apparent sizes, and complex sea background. We propose **IS-YOLO**, a YOLOv7-based framework designed to enhance feature extraction and multi-scale fusion for **small IR targets**. The backbone replaces E-ELAN blocks with **E-C2G (ELAN-2Conv + GhostConv)** and integrates a **ResC3** unit to reduce redundancy and improve gradient flow. The neck adopts **MPPELAN**, a max-pooling-pyramid ELAN variant that strengthens multi-scale context aggregation. On our FLIR-captured maritime IR dataset and public benchmarks, IS-YOLO improves precision/recall and AP metrics over state-of-the-art detectors. :contentReference[oaicite:3]{index=3}

---

## Method
### Architecture at a Glance
- **Backbone:** E-C2G (ELAN-2Conv + GhostConv) + **ResC3**  
  *Motivation:* richer features with shorter/longer gradient paths balanced; reduce parameter redundancy while preserving representational power. :contentReference[oaicite:4]{index=4}
- **Neck:** **MPPELAN** (Max-Pooling Pyramid-ELAN)  
  *Motivation:* robust **multi-scale fusion** using stacked conv + multi-kernel max-pooling, concatenation, and refinement conv to capture local details and global context for small targets. :contentReference[oaicite:5]{index=5}
- **Head:** YOLOv7 detection head (multi-scale outputs). :contentReference[oaicite:6]{index=6}

> **Why it helps small targets:** small ships often appear with weak texture and are easily confused by sea clutter; **E-C2G + MPPELAN** increases discriminative features and stabilizes small-object cues across scales. :contentReference[oaicite:7]{index=7}

<p align="center">
  <!-- Replace with your architecture figure if you want -->
  <img src="./architecture.png" alt="IS-YOLO Architecture" width="55%"/>
</p>

---

## Dataset
- **Source:** Yeungnam University maritime IR small-ship dataset captured with **FLIR T620**.  
- **Format:** YOLO labels.  
- **Split:** **1,370** training images, **120** validation images.  
- **Scene diversity:** multiple regions; varied target counts (1–9 per image); heterogeneous sea backgrounds. :contentReference[oaicite:8]{index=8}

> **Public benchmarks used for comparison:** **SIRST-v2** (512 images; urban/interference-heavy) and **IRSTD-1k** (1,000 IR images; drones/creatures/vessels/vehicles across diverse scenes). :contentReference[oaicite:9]{index=9}

> **Add your result photos:**  
> Put qualitative examples in `./figures/results/` and reference them below.

<p align="center">
  <!-- Example slots for your qualitative results -->
  <img src="./figures/results/sample_01.jpg" alt="Qualitative Result 1" width="45%"/>
  <img src="./figures/results/sample_02.jpg" alt="Qualitative Result 2" width="45%"/>
</p>

---

## Training & Evaluation
- **Environment (paper setup):** Python 3.8.19, PyTorch 2.2.1, CUDA 11.2; Ryzen 9 7950X, RTX 4090; Ubuntu 23.04.  
- **Typical config:** image size 640×640, batch size 1, epochs 50, Adam optimizer, LR 0.01. :contentReference[oaicite:10]{index=10}
- **Metrics:** AP@0.5, AP@0.5:0.95, Params, Model Size, GFLOPs, Inference Time. :contentReference[oaicite:11]{index=11}

> **Note:** This repository currently hosts **paper/patent/slides & results**. If licensing allows in the future, we plan to release a minimal **inference script** and **sample IR images** for reproducibility (without disclosing restricted data).

---

## Results

### Main (Ablations, Own IR Dataset)
| Model Variant | AP@0.5 | AP@0.5:0.95 | Params (M) | Model Size (MB) |
|---|---:|---:|---:|---:|
| YOLOv7 (ELAN+SPPCSPC) | 86.5 | 34.9 | 36.4 | 71.3 |
| ELAN + MPPELAN | 84.6 | 35.5 | 28.8 | 59.6 |
| E-C2G + SPPCSPC | 83.9 | 33.5 | 40.4 | 81.5 |
| E-C2G + SPPELAN | 84.1 | 35.6 | **32.8** | **63.1** |
| **IS-YOLO (E-C2G + MPPELAN)** | **88.9** | **38.3** | **32.8** | **63.1** |
<sub>Numbers from the paper’s ablation table. :contentReference[oaicite:12]{index=12}</sub>

### Cross-Method Comparison (Own IR Dataset)
| Method | AP@0.5 | AP@0.5:0.95 | Params (M) | Size (MB) |
|---|---:|---:|---:|---:|
| YOLOv7 | 86.5 | 34.9 | 36.4 | 71.3 |
| Faster R-CNN | 84.0 | 30.9 | – | 533 |
| FCOS | 77.3 | 28.8 | – | 244 |
| RetinaNet | 79.6 | 29.1 | – | 278 |
| YOLOv8-l | 84.9 | 35.2 | 43.6 | 83.7 |
| YOLOv9 | 83.9 | 32.0 | 60.4 | 116 |
| **IS-YOLO** | **88.9** | **38.3** | **32.8** | **63.1** |
<sub>IS-YOLO outperforms across AP metrics with competitive complexity. :contentReference[oaicite:13]{index=13}</sub>

### Public Benchmarks
**SIRST-v2**
- YOLOv7: P 89.1 / R 65.7 / AP@0.5 71.6 / AP@0.5:0.95 30.7  
- **IS-YOLO:** P 88.3 / **R 77.4** / **AP@0.5 80.3** / **AP@0.5:0.95 32.6** :contentReference[oaicite:14]{index=14}

**IRSTD-1k**
- YOLOv7: P 85.3 / R 78.3 / AP@0.5 82.2 / AP@0.5:0.95 32.5  
- **IS-YOLO:** P 84.6 / **R 79.7** / **AP@0.5 82.8** / **AP@0.5:0.95 33.6** :contentReference[oaicite:15]{index=15}

---


## How to Use This Folder
```

ai-ml-portfolio/Computer Vision/Small Ship Detection (IS-YOLO)/
├─ README.md              # this file
├─ paper.pdf              # add your paper here
├─ patent.pdf             # add your patent here
├─ slides.pdf             # (optional) add your slides
├─ architecture.png    # (optional) model diagram
└─ results/
├─ sample_01.jpg    # qualitative results (add more as needed)
└─ sample_02.jpg

````

---

## Citation
If you use IS-YOLO or this dataset description, please cite the paper:

```bibtex
@article{Firdiantika2024ISYOLO,
  title   = {IS-YOLO: A YOLOv7-based Detection Method for Small Ship Detection in Infrared Images With Heterogeneous Backgrounds},
  author  = {Indah Monisa Firdiantika and Sungho Kim},
  journal = {International Journal of Control, Automation, and Systems},
  year    = {2024},
  volume  = {22},
  number  = {11},
  pages   = {3295--3302},
  doi     = {10.1007/s12555-024-0044-8}
}
````

---

## Acknowledgments

This work was supported by the **National Research Foundation of Korea (MSIT)** (No. RS-2023-00219725) and **Yeungnam University Research Grants (2024)**. 


```
