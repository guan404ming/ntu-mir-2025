# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is NTU MIR 2025 Homework 1 - a music information retrieval assignment focused on artist recognition using the Artist20 dataset. The project implements both traditional machine learning (Task 1) and deep learning (Task 2) approaches for classifying music tracks by artist.

### Implementation Status

- ✅ Task 1: Traditional ML (SVM, Random Forest, k-NN) - 57.14% top-1, 78.79% top-3
- ✅ Task 2: Deep Learning (PANNs-based) - 60.17% top-1, 83.12% top-3
- ✅ Test predictions generated: `r14921046.json`
- ✅ Reports and visualizations completed

## Dataset Structure

The Artist20 dataset contains:

- 20 artists, 120 albums, 1413 tracks total
- Train/Val/Test split: 949/231/233 tracks
- Audio format: 16kHz mono MP3 files
- Data located in `data/artist20/`

Key files:

- `data/artist20/train.json` - Training set file paths and labels
- `data/artist20/val.json` - Validation set file paths and labels
- `data/artist20/test/` - Test audio files (001.mp3 to 233.mp3)
- `data/artist20/train_val/` - Training and validation audio files organized by artist/album

## Project Files

### Python Scripts

- **task1_preprocessing.py** - Extract audio features for traditional ML (MFCC, mel spectrogram, spectral features)
- **task1_train.py** - Train traditional ML models (SVM, Random Forest, k-NN) with grid search
- **task1_gen_report.py** - Generate Task 1 visualizations (confusion matrix, model comparison)
- **task2_train.py** - Train PANNs-based deep learning model with data augmentation
- **task2_inference.py** - Generate test set predictions (outputs `r14921046.json`)
- **task2_gen_report.py** - Generate Task 2 visualizations (confusion matrix, training curves, architecture diagram)
- **count_score.py** - Evaluate predictions against ground truth

### Output Files

- **r14921046.json** - Test predictions for submission (233 tracks, top-3 predictions each)
- **results/** - Saved models, preprocessed features, training logs
- **assets/** - Generated charts and confusion matrices
- **report.md** - Marp presentation with full project report

### Requirements

See `requirements.txt` for dependencies:

- librosa==0.11.0 (audio feature extraction)
- torch>=2.0.0, torchaudio==2.8.0 (deep learning)
- scikit-learn==1.7.2 (traditional ML)
- panns-inference==0.1.1 (pretrained audio model)
- matplotlib, seaborn (visualization)

## Common Commands

### Setup

```bash
# Download and extract dataset
bash get_dataset.sh

# Install dependencies
pip install -r requirements.txt
```

### Task 1: Traditional ML

```bash
# Extract features and preprocess
python task1_preprocessing.py

# Train models with grid search
python task1_train.py

# Generate report visualizations
python task1_gen_report.py
```

### Task 2: Deep Learning

```bash
# Train PANNs-based model (150s audio, 100 epochs)
python task2_train.py

# Generate test predictions
python task2_inference.py

# Generate report visualizations
python task2_gen_report.py
```

### Evaluation

```bash
# Evaluate predictions against ground truth
python count_score.py ./data/test_ans.json ./r14921046.json
```

## Output Format Requirements

Your model predictions must follow the exact format in `test_pred_example.json`:

```json
{
    "001": ["top1_pred", "top2_pred", "top3_pred"],
    "002": ["top1_pred", "top2_pred", "top3_pred"],
    ...
    "233": ["top1_pred", "top2_pred", "top3_pred"]
}
```

Where each test file (001.mp3 to 233.mp3) maps to an array of top-3 artist predictions.

## Implementation Details

### Task 1: Traditional ML

**Features Extracted:**

- MFCC: 20 coefficients + Delta + Delta-Delta (60 features)
- Mel Spectrogram: 128 bins with statistical aggregation
- Spectral Features: Centroid, Bandwidth, Rolloff, Contrast (6 bands)
- Energy: RMS Energy
- Rhythm: Zero Crossing Rate, Tempo
- Chroma-like: First 12 mel bins
- Total: 400+ features → reduced to 100 via SelectKBest

**Preprocessing:**

- RobustScaler for outlier resistance
- Statistical aggregation (mean, std, max, min)
- Feature selection (SelectKBest, f_classif)

**Models:**

- SVM: RBF/Linear/Poly kernels, grid search (C, gamma, degree)
- Random Forest: 100-300 trees
- k-NN: k=3-15 with distance weighting

**Results:**

- Best: SVM with 57.14% top-1, 78.79% top-3

### Task 2: Deep Learning

**Architecture:**

- Feature extractor: PANNs (Pretrained Audio Neural Networks) - frozen
- Embedding: 2048-dimensional features
- Classifier: 4-layer MLP (2048→1024→512→256→20)
- Regularization: Dropout (0.3, 0.4, 0.3, 0.2) + BatchNorm1d

**Training:**

- Audio: 150 seconds at 16kHz
- Data augmentation: Random cropping, noise, time stretch
- Optimizer: Adam with weight decay (1e-4)
- Learning rate: 0.005 with ReduceLROnPlateau
- Early stopping: Based on validation performance
- Epochs: 100 maximum

**Results:**

- Validation: 60.17% top-1, 83.12% top-3
- Improvement over SVM: +3.03% top-1, +4.33% top-3

## Important Constraints

- **DO NOT** use audio files in the `test/` folder for training
- Use `train.json` for training and `val.json` for validation only
- Final evaluation uses `count_score.py` which calculates top-1 and top-3 accuracy
- Test predictions must follow exact format: `{"001": ["artist1", "artist2", "artist3"], ...}`

---

## Original Assignment Requirements (Reference)

### Dataset: Artist20

- 20 artists, 120 albums, 1413 tracks
- Train/Validation/Test: 949/231/233 tracks
- Album-level split: 4/1/1
- Sample rate: 16kHz, mono MP3, full song length

### Task 1: Traditional ML

- Extract audio features (MFCC, spectral, rhythm, etc.)
- Apply standardization, pooling, normalization
- Train k-NN, SVM, or Random Forest
- Report: features, model, confusion matrix, top-1/top-3 accuracy

### Task 2: Deep Learning

- Implement deep learning model
- Cite any referenced work
- Report: architecture, confusion matrix, top-1/top-3 accuracy
- Submit test predictions to NTUCool

### Reference Methods

- Music classification: [sota-music-tagging-models](https://github.com/minzwon/sota-music-tagging-models)
- Singer recognition: [music-artist-classification-crnn](https://github.com/ZainNasrullah/music-artist-classification-crnn)
- Pretrained models: PANNs, self-supervised learning

### Evaluation Metrics

- Confusion matrix: Predicted vs actual labels
- Top-k accuracy: Correct if true label in top-k predictions
