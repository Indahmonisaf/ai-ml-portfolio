
### Pneumonia Detection 
**Problem (detect pneumonia from chest X-rays)**  
Binary classification of chest X-ray images into pneumonia vs. normal using transfer learning with CNNs to enable faster, automated screening.

**Dataset / Source (splits)**  
Kaggle “Chest X-Ray Images (Pneumonia)” (Guangzhou Women & Children’s Medical Center, pediatric 1–5 y.o.). Public dataset with expert-validated labels.  
Original folders: Train 5,216 (3,875 pneumonia / 1,341 normal) and Test 624 (390 pneumonia / 234 normal). Authors then formed 5 folds (80/20) and, for each fold, evaluated on 50 images/class.

**Method / Model (training details & interpretability)**  
CNN (transfer learning): ResNet-50 and VGG-16, input resized to 128×128, trained for 5 and 10 epochs.  
Metrics computed: Accuracy, Precision, Recall (Sensitivity), Specificity, F1; runtime also reported. (AUC and Grad-CAM were not reported; you can add Grad-CAM in your repo for interpretability.)

**Metrics (per-fold results; highlights)**  
Overall ranges across folds: Accuracy 0.86–0.98, Precision 0.81–1.00, Recall 0.80–0.98.  
Best fold: ResNet-50 (10 epochs) reached Accuracy 0.98, Sensitivity 0.98, Specificity 1.00.  
Latency: ResNet-50 testing is faster than VGG-16 (e.g., ~12–15 s vs. 20–174 s per test batch in tables).

**Result (paper title + authors)**  
“Pneumonia Detection in Chest X-ray Images Using Convolutional Neural Network.”  
Authors: Indah Monisa Firdiantika, Yessi Jusman. Venue: AIP Conference Proceedings (ICITAMEE 2021).
