
# Box Recognition Method and Computing Device (EGCY-Net)

**Paper Title + Authors**  
“EGCY-Net: An ELAN and GhostConv-Based YOLO Network for Stacked Packages in Logistic Systems.”  
Authors: Indah Monisa Firdiantika, Seongryeong Lee, Chaitali Bhattacharyya, Yewon Jang, Sungho Kim.  
Journal: Applied Sciences, 14(7):2763, 26 Mar 2024. doi: 10.3390/app14072763.

**Problem & Industrial Use Case**  
Automation of recognizing the type & quantity of boxes in factories/logistics to support AGV flow (pick–stick barcode–store), replacing manual inspection which is slow/less consistent. Challenges: similar box types (small height differences), varying environments, and the need to run on resource-constrained embedded devices.

**Dataset/Source (translated)**  
- **Box availability** (classification “box” vs “no box”): internal, from factory videos (Pyeong Hwa Automotive), ROI cropped → 1,200 box + 1,200 no-box (JPG).  
- **Box type & quantity** (detection of 4 classes: 04, 13, A, Y): internal, phone/Arducam images & videos; high-res crops per stack → 1,258 images total; train/val/test distribution per class:  
  - 04: 420 / 80 / 12  
  - 13: 420 / 80 / 34  
  - A: 420 / 80 / 40  
  - Y: 420 / 80 / 22.

**Method/Model (translated)**  
- **Stage 1 – Box availability:** MobileNet (ROI classification: box vs no-box).  
- **Stage 2 – Box type & counting:** EGCY-Net (YOLO-based) with:  
  - **CGStack** (Conv-GhostConv-Conv) for parameter efficiency & spatial features,  
  - **EGCNet** (ELAN + CGStack) as feature aggregation module (backbone/neck),  
  - 3-scale detection head.  
- **Post-process:** count valid boxes based on min size & score > 0.5, then aggregate counts per class.

**Reported Metrics (translated)**  
Used: Precision, Recall, mAP@0.5, mAP@0.5:0.95, number of parameters, ONNX size, and inference time (s).  
SOTA comparison (YOLOv3, YOLOv5l, YOLOR, YOLOv7x, YOLOv7) on the internal dataset, 20 epochs (except EGCY-Net continued to 300 epochs for final improvements).

**Main Results (4-class detection, internal dataset) (translated)**  
mAP@0.5:0.95 (All): **87.1%** (highest).  
Per-class mAP@0.5:0.95: 04 **86.5%**, 13 **87.3%**, A **86.6%**, Y **88.6%**.  
Parameters: **30 M** (smallest).  
ONNX size: **111 MB** (smallest).  
Inference time: **5.96 s** (fastest in comparison table).  
Precision/Recall (All): **99.6% / 99.9%**.  
EGCY-Net outperforms YOLOv3/5l/YOLOR/YOLOv7/x in aggregate accuracy while being lighter and faster.

**Implementation & Device Latency (translated)**  
Training on a workstation (RTX 4090); deployed on **NVIDIA Jetson Nano + Arducam IMX477**; model converted PyTorch → ONNX for cross-device inference.
