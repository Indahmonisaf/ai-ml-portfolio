
# Skin Cancer Classification (CNN)

**Problem (multi-class skin-lesion classification)**  
Early, automated classification of dermatoscopic images into benign vs. malignant lesion types; paper compares MLP, a custom CNN, and transfer-learned VGG-16 on HAM10000 to balance accuracy and practical (testing) latency.

**Paper Title + Authors**  
“Performance of Multi Layer Perceptron and Deep Neural Networks in Skin Cancer Classification.”  
Authors: Yessi Jusman, Indah Monisa Firdiantika, Dhimas Arief Dharmawan, Kunnu Purwanto.  
Venue: IEEE 3rd Global Conference on Life Sciences and Technologies (LifeTech 2021).

**Dataset / Source**  
HAM10000 (public): 10,015 dermatoscopic images collected over ~20 years from Austria and Australia; seven lesion classes (AKIEC, BCC, BKL, DF, NV, VASC, MEL). The paper evaluates five binary tasks: MEL–NV, AKIEC–BKL, BCC–VASC, MEL–AKIEC, BKL–BCC.

**Method / Model**  
- Baselines: Multi-Layer Perceptron (MLP), custom CNN (4 conv + 4 max-pool + dense layers).  
- Transfer learning: VGG-16 (pretrained on ImageNet) with input resized to 128×128 for tractable training time.  
- Implementation: Python/Keras; comparison by classification accuracy and train/test time. (No explicit class-imbalance handling is reported in the paper.)

**Metrics**  
Reported: Accuracy (train/val/test) and computational time (training/testing).  

**Key Results (Testing Accuracy, best per task)**  
- MEL vs NV: 84.89% (VGG-16 TL).  
- AKIEC vs BKL: 87.12% (custom CNN).  
- BCC vs VASC: 97.56% (VGG-16 TL).  
- MEL vs AKIEC: 80.00% (custom CNN).  
- BKL vs BCC: 86.67% (custom CNN).  
Overall, CNNs (especially VGG-16 TL) outperform MLP in accuracy.

**Latency / Compute (Testing Time; lower is better)**  
CNNs and VGG-16 have much faster testing than MLP despite longer training. Example—MEL vs NV: CNN 35.77s, VGG-16 TL 39.51s, MLP 199.76s (on the paper’s test set and hardware). Similar trends hold across tasks.
