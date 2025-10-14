
# AI Vision for Overlapped Steel Sheet Detection

**Problem & Industrial Context**  
 Detect overlapped steel sheets (double-feed) on a feeder/stacker line before press/cutting/stamping to prevent machine jams, die damage, and quality defects—while keeping false alarms low so the line doesn’t stop unnecessarily.

**Dataset / Source**  
Internal (custom) — RGB images with polygon masks (COCO-style) for sheet instances; split into train/val (counts not printed in the current notebook).  
Camera — standard RGB imaging (not line-scan) based on the notebook pipeline and file structure.

**Method / Model**  
- Instance Segmentation: Mask R-CNN (TensorFlow 2.5 / Keras), fine-tuning detection heads.  
- Post-processing for “overlap” decision:  
  - Compute mask intersections pairwise.  
  - Mark “overlapped” if intersection ratio (e.g., IoU or intersection-over-smaller-area) exceeds a threshold.  
  - Optional filters: minimum area, confidence threshold, and proximity constraints to suppress edge noise.

**Augmentation (recommended for robustness)**  
Horizontal/vertical flip, small rotations/translations, brightness/contrast jitter, light Gaussian noise, and mild motion blur to mimic conveyor movement.

**Metrics**  
Segmentation/Detection: IoU (per instance), Precision/Recall/F1, and/or mAP (mask).  
Overlap decision quality: False Alarm Rate (FAR) and Miss Rate on overlapped vs non-overlapped events.  
Latency: end-to-end inference time per frame (ms) on the target device (IPC/edge GPU).  
(The current notebook doesn’t print final metric numbers; add an evaluation cell to compute IoU/F1/FAR from predictions vs. ground truth.)

**Result (project title + contributors)**  
Project Title: AI Vision for Overlapped Steel Sheet Detection  
Status: Internal project using Mask R-CNN with overlap post-processing logic.  
Contributors: (Add your name and team members here.)

**Implementation Notes**  
Notebook uses a COCO-style loader, custom config, and head-only fine-tuning.  
For production: consider a lighter model (e.g., YOLOv8-Seg / RT-DETR-Seg) for lower latency; calibrate overlap thresholds across product types/thickness; add confidence/area gating to reduce FAR; and integrate a fail-safe (e.g., physical sensor) when model confidence is low.
