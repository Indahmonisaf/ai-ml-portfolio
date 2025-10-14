# Pneumonia Detection (CXR, CNN)

**Problem**  
Binary classification of pediatric chest X-rays: pneumonia vs normal via transfer learning.

**Dataset**  
Kaggle “Chest X-Ray Images (Pneumonia)” (GWCMC).  
Original: Train 5,216 (3,875 P / 1,341 N), Test 624 (390 P / 234 N).  
5-folds (80/20); per fold evaluate 50 imgs/class.

**Methods**  
**ResNet-50** and **VGG-16**, input 128×128, trained 5 & 10 epochs. Metrics: Acc, Precision, Recall, Specificity, F1; runtime.

**Highlights**  
Acc range **0.86–0.98**, Precision **0.81–1.00**, Recall **0.80–0.98**.  
Best fold: ResNet-50 (10 ep) **Acc 0.98**, **Sens 0.98**, **Spec 1.00**.  
ResNet-50 testing faster than VGG-16.

**Reference**  
“**Pneumonia Detection in Chest X-ray Images Using Convolutional Neural Network**.”  
*Indah Monisa Firdiantika, Yessi Jusman.* **AIP Conf. Proc. (ICITAMEE 2021)**.

## How to Run
- `notebooks/pneumonia_cxr.ipynb` with train/eval.
- (Optional) add Grad-CAM in `interpretability/`.

## Results
- Fold metrics in `reports/`.
- Heatmaps in `results/`.

