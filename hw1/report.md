---
marp: true
theme: default
paginate: true
backgroundColor: white
style: @import url('https://unpkg.com/tailwindcss@^2/dist/utilities.min.css');
---

# NTU MIR 2025 - Homework 1
## Task 1: Traditional Machine Learning for Artist Recognition

**Student:** 邱冠銘
**Student ID:** R14921046

---

## Audio Features Extracted

<div class="grid grid-cols-2 gap-4">
<div>

### Core Features
- **MFCC Features:** 20 coefficients + Delta (velocity) + Delta-Delta (acceleration)
- **Mel Spectrogram:** 128 mel frequency bins with statistical aggregation
- **Spectral Features:** Centroid, Bandwidth (frequency-weighted), Rolloff (mel energy sum)

</div>
<div>

### Additional Features
- **Energy Features:** RMS Energy with temporal statistics
- **Rhythm Features:** Zero Crossing Rate, Tempo (from onset strength detection)
- **Advanced Features:**
  - Chroma-like features (first 12 mel bins)
  - Spectral Contrast (6 frequency bands, peak-to-valley ratios)

</div>
</div>

---

## Preprocessing Pipeline

### Data Processing Steps
- **RobustScaler** for outlier resistance
- **Feature Selection** (SelectKBest, top 100 features)
- **Statistical Aggregation** (mean, std, max, min over time)

### Feature Engineering
- **400+ dimensional features** → reduced to 100 via feature selection
- **GPU-accelerated extraction** with PyTorch for efficiency
- **Robust preprocessing** to handle NaN/inf values

---

## Traditional ML Models

### Models Implemented
- **SVM:** RBF kernel with grid search (C, γ hyperparameters)
- **Random Forest:** Grid search over 100-300 trees with hyperparameter tuning
- **k-NN:** Grid search over k=3-15 with distance weighting

### Training Process
- **5-fold cross-validation** for hyperparameter tuning
- **Grid search** for optimal hyperparameters
- **Stratified validation** to ensure balanced evaluation

---

## Model Comparison



<div class="grid grid-cols-2 gap-4">
<div>

| Model | Top-1 Accuracy | Top-3 Accuracy |
|-------|----------------|----------------|
| **SVM** | **57.14%** | **78.79%** |
| Random Forest | 46.32% | 72.73% |
| k-NN | 37.23% | 61.04% |

</div>
<div>

![](results/task1/model_comparison.png)

</div>
</div>

---

## Confusion Matrix - SVM

![bg right:50%](results/task1/confusion_matrix_svm.png)

- Strong diagonal indicates good overall classification
- Some artists more easily distinguishable than others
- Confusion often occurs between similar music styles

---

## Key Findings
- **Best Model:** SVM with 57.14% top-1 and 78.79% top-3 accuracy
- **Feature diversity** and **proper preprocessing** are crucial
- **SVM with RBF kernel** works well for high-dimensional audio features