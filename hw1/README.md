# NTU MIR 2025 - Homework 1

Artist Recognition using Traditional ML and Deep Learning

**Student:** 邱冠銘
**Student ID:** R14921046

## Overview

This project implements artist recognition on the Artist20 dataset using both traditional machine learning and deep learning approaches.

## Quick Start

### Setup
```bash
pip install -r requirements.txt
bash get_dataset.sh
```

### Inference (Generate Test Predictions)
```bash
# Outputs: r14921046.json
python task2_inference.py 

# For specific output path, test directory, model path, and train JSON path
python task2_inference.py \
    --output r14921046.json \
    --test_dir data/artist20/test \
    --model_path assets/best_panns_model.pth \
    --train_json data/artist20/train.json
```

### Training

**Task 1 - Traditional ML:**
```bash
python task1_preprocessing.py  # Extract features
python task1_train.py          # Train SVM, Random Forest, k-NN
python task1_gen_report.py     # Generate visualizations
```

**Task 2 - Deep Learning:**
```bash
python task2_train.py          # Train PANNs-based model
python task2_gen_report.py     # Generate visualizations
```


### Evaluation
```bash
python count_score.py ./data/test_ans.json ./r14921046.json
```

## Results

| Approach | Model | Top-1 Accuracy | Top-3 Accuracy |
|----------|-------|----------------|----------------|
| Traditional ML | SVM | 57.14% | 78.79% |
| Deep Learning | PANNs-based | 60.17% | 83.12% |

## Project Structure

```
├── task1_preprocessing.py    # Feature extraction for traditional ML
├── task1_train.py            # Train traditional ML models
├── task1_gen_report.py       # Generate Task 1 visualizations
├── task2_train.py            # Train deep learning model
├── task2_inference.py        # Generate test predictions
├── task2_gen_report.py       # Generate Task 2 visualizations
├── report.md                 # Detailed report (Marp presentation)
└── assets/                   # Generated charts and figures
```

## Documentation

See [CLAUDE.md](CLAUDE.md) and [report.md](report.md) for detailed project documentation.