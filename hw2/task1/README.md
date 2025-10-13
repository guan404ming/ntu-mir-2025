# Task 1: Music Retrieval

This directory implements Task 1 of the homework, which involves retrieving similar reference music for each target track and evaluating the results using multiple metrics.

## Overview

The task performs:
1. **Retrieval**: Find the most similar reference tracks for each target music using audio embeddings
2. **Evaluation**: Measure quality using CLAP similarity, Melody accuracy, and Audiobox Aesthetics

## Project Structure

```
task1/
├── encoders/           # Audio embedding encoders
│   ├── base_encoder.py           # Base encoder interface
│   ├── clap_encoder.py          # CLAP encoder
│   └── stable_audio_encoder.py  # Stable Audio VAE encoder
├── metrics/            # Evaluation metrics
│   ├── clap_metric.py           # CLAP similarity
│   ├── melody_metric.py         # Melody accuracy
│   └── aesthetics_metric.py     # Audiobox aesthetics
├── utils/              # Utility functions
├── retrieval.py        # Music retrieval system
└── main.py            # Main execution script
```

## Installation

All dependencies are already installed via `uv` in the parent directory. Key dependencies:
- `torch`, `torchaudio` - Deep learning and audio processing
- `librosa` - Audio feature extraction
- `scipy` - Signal processing
- `laion-clap` - CLAP audio-text model
- `numpy` - Numerical operations

## Usage

### Basic Usage

Run retrieval with default settings (CLAP encoder):

```bash
cd task1
python main.py
```

### Advanced Options

```bash
python main.py \
    --encoder clap \
    --reference-dir ../data/referecne_music_list_60s \
    --target-dir ../data/target_music_list_60s \
    --top-k 5 \
    --output-dir results \
    --device cuda \
    --use-dummy-aesthetics
```

### Arguments

- `--encoder`: Audio encoder to use (`clap`, `music2latent`, `muq`)
- `--reference-dir`: Directory with reference music files
- `--target-dir`: Directory with target music files
- `--top-k`: Number of similar tracks to retrieve (default: 5)
- `--output-dir`: Directory to save results (default: `results`)
- `--cache-dir`: Directory for caching embeddings (default: `cache`)
- `--device`: Device to use (`cuda` or `cpu`, default: `cuda`)
- `--use-dummy-aesthetics`: Use dummy aesthetics metric if audiobox unavailable
- `--target-duration`: Trim target audio to this duration for melody accuracy (seconds)

## Output

The script generates two JSON files in the output directory:

### 1. Retrieval Results (`retrieval_results_<encoder>.json`)

Maps each target track to its top-k most similar reference tracks:

```json
{
  "target_song.wav": [
    {
      "reference": "reference_song_1.mp3",
      "similarity": 0.8542
    },
    ...
  ]
}
```

### 2. Evaluation Results (`evaluation_results_<encoder>.json`)

Detailed metrics for each target track:

```json
{
  "target_song.wav": {
    "retrieved_track": "reference_song_1.mp3",
    "retrieval_similarity": 0.8542,
    "clap_similarity": 0.7821,
    "melody_accuracy": 0.1234,
    "aesthetics": {
      "ce": 0.756,
      "cu": 0.823,
      "pc": 0.691,
      "pq": 0.778
    }
  }
}
```

## Evaluation Metrics

### 1. CLAP Similarity
- Cosine similarity between CLAP audio embeddings
- Range: [-1, 1], higher is more similar
- Measures overall audio similarity

### 2. Melody Accuracy
- Frame-by-frame chromagram matching
- Range: [0, 1], higher is better
- Extracts one-hot melody representation and compares pitch classes
- Based on `data/Melody_acc.py`

### 3. Audiobox Aesthetics
Four quality dimensions:
- **CE (Content Enjoyment)**: How enjoyable the content is
- **CU (Content Usefulness)**: How useful/appropriate the content is
- **PC (Production Complexity)**: Complexity of the production
- **PQ (Production Quality)**: Technical quality of the production

## Encoders

### CLAP (Default)
- Contrastive Language-Audio Pretraining
- Pre-trained on large-scale audio-text pairs
- Good for general audio similarity
- Model: `music_audioset_epoch_15_esc_90.14.pt`
- **Status**: ✅ Working

### Music2Latent
- Consistency Autoencoder for latent audio compression
- Developed by Sony CSL Paris
- Encodes 44.1 kHz audio into 64-channel latent representations
- **Status**: ⚠️ CUDA compatibility issues (use CPU mode with `--device cpu`)
- **Note**: Works on CPU but experiences segmentation faults on CUDA

### MuQ (Music Quantization)
- Self-supervised music representation learning with Mel Residual Vector Quantization
- Developed by Tencent AI Lab
- 310M parameters, trained on large-scale music data
- Encodes 24 kHz audio using Conformer architecture
- **Status**: ✅ Working on both CPU and CUDA

## Caching

The system caches audio embeddings to speed up repeated runs:
- Cache location: `cache/` directory
- Separate cache per encoder
- Automatically recomputes if reference files change

## Notes

1. **Reference directory typo**: The original dataset has a typo `referecne_music_list_60s` (note the misspelling)
2. **Dummy aesthetics**: If Meta Audiobox is not installed, use `--use-dummy-aesthetics` flag for testing
3. **GPU usage**: CUDA is used by default if available
4. **Melody trimming**: Use `--target-duration` to trim target audio if generated audio is shorter (e.g., 47 seconds for MuseControlLite)

## Example Run

```bash
# Run retrieval with CLAP encoder
python main.py --encoder clap --device cuda

# Check results
cat results/retrieval_results_clap.json
cat results/evaluation_results_clap.json
```
