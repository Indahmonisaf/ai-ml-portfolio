
# Skin Cancer Classification (CNN)

**Problem**  
Classify dermatoscopic images (benign vs malignant types). Compare MLP, custom CNN, and VGG-16 transfer learning on **HAM10000**.

**Dataset**  
HAM10000 (10,015 images; 7 classes: AKIEC, BCC, BKL, DF, NV, VASC, MEL). Paper evaluates 5 binary tasks (e.g., MEL–NV, AKIEC–BKL).

**Methods**
- Baselines: **MLP**, **custom CNN** (4 conv + 4 max-pool + dense).
- Transfer learning: **VGG-16** (ImageNet), input 128×128.
- Impl: Python/Keras; compare accuracy & train/test time.

**Key Results (Testing Acc, best per task)**
- MEL vs NV **84.89%** (VGG-16 TL)  
- AKIEC vs BKL **87.12%** (custom CNN)  
- BCC vs VASC **97.56%** (VGG-16 TL)  
- MEL vs AKIEC **80.00%** (custom CNN)  
- BKL vs BCC **86.67%** (custom CNN)

**Latency (Testing time)**  
CNN/VGG-16 faster at test time than MLP (e.g., MEL vs NV: CNN 35.77s, VGG-16 39.51s, MLP 199.76s).

**Reference**  
“**Performance of Multi Layer Perceptron and Deep Neural Networks in Skin Cancer Classification**.”  
*Yessi Jusman, Indah Monisa Firdiantika, Dhimas Arief Dharmawan, Kunnu Purwanto.* **IEEE LifeTech 2021**.

## How to Run
- `notebooks/skin_cancer_cnn.ipynb` (train/eval/plot)
- Add optional ROC-AUC/Sensitivity/Specificity computation.

## Results
- Accuracy/time tables in `reports/metrics.md`.
- Confusion matrices & sample predictions in `results/`.
