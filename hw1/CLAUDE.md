# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is NTU MIR 2025 Homework 1 - a music information retrieval assignment focused on artist recognition using the Artist20 dataset. The project requires implementing both traditional machine learning and deep learning approaches for classifying music tracks by artist.

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

## Common Commands

### Dataset Setup
```bash
# Download and extract dataset
bash get_dataset.sh
```

### Model Evaluation
```bash
# Evaluate predictions against ground truth
python count_score.py ./test_ans.json ./test_pred.json
# Or using the data directory version
python data/count_score.py ./data/test_ans.json ./data/test_pred.json
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

## Task Requirements

### Task 1: Traditional ML Model
- Use traditional ML algorithms (k-NN, SVM, Random Forest)
- Extract audio features using libraries like Librosa or Torchaudio
- Apply standardization, pooling, and normalization
- Report validation results with confusion matrix, top-1 and top-3 accuracy

### Task 2: Deep Learning Model
- Implement a deep learning model from scratch
- Submit test set predictions to NTUCool
- Report validation results with confusion matrix, top-1 and top-3 accuracy

## Important Constraints

- **DO NOT** use audio files in the `test/` folder for training
- Use `train.json` for training and `val.json` for validation only
- Final evaluation uses `count_score.py` which calculates top-1 and top-3 accuracy

## Recommended Libraries

- Audio processing: Librosa, Torchaudio
- ML models: Scikit-learn
- Deep learning: PyTorch, TensorFlow