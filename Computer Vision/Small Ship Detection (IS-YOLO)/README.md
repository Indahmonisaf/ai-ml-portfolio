# Small Ship Detection in Infrared Images (IS-YOLO)

**Goal & Context**  
Detect small ships in IR imagery under heterogeneous sea/urban backgrounds where low signal-to-clutter, blurred contours, specular reflections, and scale variance make detection difficult. IS-YOLO (YOLOv7-based) improves accuracy and real-time performance for maritime applications.

**Publication**  
“**IS-YOLO: A YOLOv7-based Detection Method for Small Ship Detection in Infrared Images With Heterogeneous Backgrounds**.”  
*Indah Monisa Firdiantika, Sungho Kim.* **IJCAS**, 22(11): 3295–3302 (2024). DOI: 10.1007/s12555-024-0044-8.

**Datasets**
- **Yeungnam Univ. IR Small Ships** (authors’ dataset, FLIR T620): 1,370 train / 120 val (YOLO labels).
- **SIRST v2**: 512 imgs (train 360 / val 103 / test 51), cluttered urban scenes.
- **IRSTD-1k**: 1,000 labeled IR imgs (drones/creatures/vessels/vehicles).

**Method / Model**
- **Backbone:** E-C2G (improved E-ELAN + GhostConv + ResC3) → better feature learning with efficient gradient paths.  
- **Neck:** **MPPELAN** (max-pool–enhanced SPPELAN) for robust multi-scale fusion.

**Training Setup (key)**  
Image 640×640, Adam, LR 0.01, 50 epochs, batch 1.  
Env: Python 3.8.19, PyTorch 2.2.1, CUDA 11.2 (RTX 4090).

**Metrics (authors’ dataset)**  
AP@0.5 **88.9%** · AP@0.5:0.95 **38.3%** · Params **32.8M** · Size **63.1 MB** · GFLOPs **121.2** · Inference **3.9 ms**.  
Outperforms YOLOv7/YOLOv8l/YOLOv9, Faster R-CNN, FCOS, RetinaNet.

**Metrics (public)**
- **SIRST v2:** P 88.3% | R 77.4% | AP@0.5 80.3% | AP@0.5:0.95 32.6% (vs YOLOv7 AP@0.5 71.6%).
- **IRSTD-1k:** P 84.6% | R 79.7% | AP@0.5 82.8% | AP@0.5:0.95 33.6% (slightly > YOLOv7).

**Ablation (highlight)**  
E-C2G backbone + MPPELAN neck yields best AP while keeping params/model size below YOLOv7 baseline.

## How to Run (after code is uploaded)
- `train.py` / `val.py` / `infer.py` with YAML configs  
- Weights & sample IR images placed in `assets/`  
- ONNX export command in `export_onnx.py`

## Results
- Qualitative examples: see `results/` (add sample IR frames).
- Benchmark tables: see `reports/metrics.md`.

## Notes
- Consider mixed precision and TensorRT for deployment.

