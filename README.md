# Age Group Detection: HOG+SVM vs HOG+MLP vs ResNet18

Classify faces into four age groups (**Child, Young, Middle-Aged, Senior**) and compare classical computer vision pipelines against a deep learning model. The best model (fine-tuned ResNet18) is combined with MTCNN face detection to run on unconstrained, real-world photos.




## Overview

This project implements and benchmarks three approaches to age group classification:

| Model | Features / Backbone | Classifier |

| **HOG + SVM** | Hand-crafted HOG descriptors | RBF-kernel SVM |
| **HOG + MLP** | Hand-crafted HOG descriptors | 2-hidden-layer neural network |
| **CNN (ResNet18)** | ImageNet-pretrained ResNet18 | Fine-tuned, 4-class head |

The dataset contains faces of **multiple ethnicities, ages, head poses and lighting conditions**, which makes this a hard, realistic problem. The best-performing model is wrapped in an `AgeDetection` function that finds faces in arbitrary images with **MTCNN**, crops them, and overlays a colour-coded age-group label and confidence score on each one.

## Pipeline


Image ──► MTCNN face detection (conf ≥ 0.90) ──► crop ──► resize 128×128 ──► ResNet18 ──► age group + confidence


### Data preprocessing
- All images resized to **128×128** px
- **Classical pipeline:** RGB → greyscale → HOG (8 orientations, 16×16 px/cell, 1×1 cells/block → **512-dim** vector) → `StandardScaler`
- **CNN pipeline:** ImageNet normalisation (mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`)
- **CNN augmentation:** Resize 160 → RandomCrop(128), RandomHorizontalFlip, RandomRotation(15°), ColorJitter, RandomGrayscale
- **Split:** 80/20 stratified train/validation split of the 13,300 training images; fixed test set of **850** images
- **Class imbalance:** handled with balanced class weights (SVM and CNN loss)





### Models

**HOG + SVM**: RBF kernel, `C=10`, `gamma='scale'`, `class_weight='balanced'`

**HOG + MLP**: hidden layers `(512, 256)`, ReLU, Adam, L2 `alpha=1e-3`, early stopping, `max_iter=200`

**CNN (ResNet18 transfer learning)**, two-phase training:
1. **Phase 1:** freeze backbone, train FC head only (Adam, lr `1e-3`, 5 epochs)
2. **Phase 2:** unfreeze `layer4` + FC (Adam, lr `5e-5`, up to 25 epochs, early stopping patience 10)
- `Dropout(0.5)` in the FC head, weighted cross-entropy with label smoothing `0.1`

## Results

Evaluated on the held-out test set (850 images):

| Model | Accuracy | F1 (weighted) | Inference (ms/img) | Size (MB) |

| HOG + SVM | 37.1% | 0.334 | 6.32 | 44.1 |
| HOG + MLP | 39.8% | 0.280 | 0.03 | 9.5 |
| **CNN (ResNet18)** | 39.8% | **0.411** | 5.05 | 44.8 |



**Key findings**
- ResNet18 was selected as the best model by **weighted F1 (0.411)**.
- HOG + MLP matches the CNN's accuracy but **collapses onto the majority "Young" class**, giving the lowest F1, so accuracy alone is misleading on this imbalanced data.
- Absolute scores are modest: four-way age grouping across diverse ethnicities, poses and lighting is difficult, and neighbouring groups (e.g. Young vs Middle-Aged) overlap heavily. Confusion matrices for all three models are produced by the comparison notebook.
- Running MTCNN first removes background clutter and improves prediction confidence on in-the-wild images.
- If no face is found, the image is skipped and labelled "no face detected" (there is no whole-image fallback).



## Repository structure


├── train_classical_models.ipynb   # HOG + SVM and HOG + MLP training/eval
├── train_cnn.ipynb                # ResNet18 transfer learning
├── compare_models.ipynb           # Metrics table, bar chart, confusion matrices, qualitative comparison
├── AgeDetection.ipynb             # MTCNN + ResNet18 demo on personal images
└── README.md


The notebooks expect this layout in Google Drive (not included in this repo):



CW_Folder_UG/
├── CW_Dataset/
│   ├── train/   (images + train_labels.txt)
│   └── test/    (images + test_labels.txt)
├── Models/      (hog_scaler.joblib, hog_svm.joblib, hog_mlp.joblib, cnn_resnet18.pth)
└── Personal_Dataset/   (your own photos for AgeDetection)



## Getting started

The notebooks are designed for **Google Colab** (enable a T4 GPU for CNN training).

1. Upload the folder structure above to Google Drive.
2. Set `GOOGLE_DRIVE_PATH_AFTER_MYDRIVE` in each notebook to your folder path.
3. Run in this order:
   1. `train_classical_models.ipynb`
   2. `train_cnn.ipynb`
   3. `compare_models.ipynb`
   4. `AgeDetection.ipynb`

### Dependencies


numpy  matplotlib  scikit-image  scikit-learn  joblib  tqdm
torch  torchvision  pillow  mtcnn


### Usage

python
AgeDetection('path/to/folder_of_images')


Picks 4 random images from the folder, detects faces, and displays each with a bounding box and age-group label (e.g. `Senior 72%`).

## Limitations and future work
- Test accuracy is around 40%; more data, a larger backbone, or ordinal (age-aware) loss functions could help.
- Class imbalance remains a challenge, so consider oversampling or focal loss.
- Fairness across ethnicities and lighting conditions deserves a per-group evaluation.
- Age predictions are approximate and should not be used for any consequential decision.

## References
1. He, K. et al. (2016). *Deep Residual Learning for Image Recognition.* CVPR, pp. 770–778.
2. Zhang, K. et al. (2016). *Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks.* IEEE Signal Processing Letters, 23(10), pp. 1499–1503.
3. Dalal, N. & Triggs, B. (2005). *Histograms of Oriented Gradients for Human Detection.* CVPR, pp. 886–893.
