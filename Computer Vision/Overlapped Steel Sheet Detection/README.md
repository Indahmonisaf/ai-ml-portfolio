
# AI Vision for Overlapped Steel Sheet Detection

**Problem & Industrial Context**  
Detect overlapped steel sheets (double-feed) on feeder/stacker lines to prevent jams, die damage, and defects—while keeping false alarms low.

**Data**  
Internal RGB images with polygon masks (COCO-style). Train/val split per notebook.

**Method**
- **Instance Segmentation:** Mask R-CNN (TF2/Keras), fine-tuning detection heads.
- **Overlap Decision:** pairwise mask intersections; classify “overlapped” if intersection ratio exceeds threshold. Optional filters (min area, confidence, proximity).

**Recommended Augmentations**  
Flip, small rotation/translation, brightness/contrast jitter, light Gaussian noise, mild motion blur.

**Metrics**
- Seg/Det: IoU, Precision/Recall/F1, mAP(mask).
- Overlap decision: False Alarm Rate (FAR), Miss Rate.
- Latency: end-to-end ms on target IPC/edge GPU.

**Status**  
Internal project; add evaluation cell to compute IoU/F1/FAR. Consider YOLOv8-Seg/RT-DETR-Seg for lower latency in production.

## How to Run
- `notebooks/steel_overlap.ipynb` (load COCO masks → train → infer → evaluate).
- Config & thresholds in `configs/overlap.yaml`.

## Results
- Add confusion matrix & example masks to `results/`.
