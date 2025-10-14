
# Box Recognition Method and Computing Device (EGCY-Net)

**Publication**  
“**EGCY-Net: An ELAN and GhostConv-Based YOLO Network for Stacked Packages in Logistic Systems**.”  
*Indah Monisa Firdiantika, Seongryeong Lee, Chaitali Bhattacharyya, Yewon Jang, Sungho Kim.* **Applied Sciences**, 14(7):2763 (Mar 26, 2024). doi: 10.3390/app14072763.

**Problem & Use Case**  
Automate box type & quantity recognition in factories/logistics to support AGV flow (pick–barcode–store). Challenges: visually similar types, varied environments, and embedded deployment constraints.

**Datasets (internal)**
- **Box availability** (ROI classification: box vs no-box) — 1,200 box + 1,200 no-box (JPG).
- **Box type & count** (4 classes: 04, 13, A, Y) — total 1,258 images, splits per class:  
  04: 420/80/12 · 13: 420/80/34 · A: 420/80/40 · Y: 420/80/22 (train/val/test).

**Method**
- Stage-1: **MobileNet** for availability (box vs no-box) on cropped ROI.  
- Stage-2: **EGCY-Net** (YOLO-based) with:
  - **CGStack** (Conv–GhostConv–Conv) for efficient parameters & spatial features
  - **EGCNet** (ELAN + CGStack) as aggregation (backbone/neck)
  - 3-scale detection head  
- Post-process: size & score filters, aggregate counts per class.

**Metrics (internal detection, 4 classes)**
- mAP@0.5:0.95 (All) **87.1%** (best)  
- Per-class mAP@0.5:0.95: 04 **86.5%**, 13 **87.3%**, A **86.6%**, Y **88.6%**  
- Params **30M** · ONNX **111 MB** · Inference **5.96 s** (fastest in table)  
- Precision/Recall (All): **99.6% / 99.9%**  
EGCY-Net outperforms YOLOv3/5l/YOLOR/YOLOv7/x while lighter & faster.

**Deployment**  
Trained on RTX 4090; deployed on **Jetson Nano + Arducam IMX477** via **ONNX**.

## How to Run
- Train/infer scripts + ONNX export in `scripts/`.
- Jetson deployment notes in `deploy/jetson.md`.

## Results
- Quantitative tables in `reports/metrics.md`.
- Demo frames in `results/`.
