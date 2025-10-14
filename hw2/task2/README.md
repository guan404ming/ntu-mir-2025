# Task 2: Music Generation

This directory implements Task 2 of the homework, which involves generating music similar to target tracks using text-to-music models conditioned on audio captions and extracted musical features.

## Overview

The task performs:
1. **Audio Captioning**: Use Audio Language Models (ALMs) to describe target music
2. **Feature Extraction**: Extract MIR features (melody, rhythm, chords) from target audio
3. **Music Generation**: Generate music using text-to-music models with extracted features
4. **Evaluation**: Measure quality using CLAP similarity, Melody accuracy, and Audiobox Aesthetics

## Project Structure

```
task2/
├── captioning/         # Audio captioning models
│   ├── base_captioner.py         # Base captioner interface
│   └── qwen_captioner.py         # Qwen-Audio captioner
├── generation/         # Music generation models
│   ├── base_generator.py         # Base generator interface
│   └── musicgen_generator.py     # MusicGen generator
├── features/           # MIR feature extraction
│   ├── melody_extractor.py       # Melody/pitch extraction
│   ├── rhythm_extractor.py       # Tempo/beat extraction
│   └── chord_extractor.py        # Chord progression extraction
├── metrics/            # Evaluation metrics (from task1)
├── utils/              # Utility functions
├── cache/              # Cached captions
├── results/            # Output results and generated audio
└── run_generation.py   # Main execution script
```

## Installation

All dependencies are managed via `uv` in the parent directory. New dependencies for Task 2:
- `transformers` - Hugging Face transformers for Qwen-Audio
- `accelerate` - Model parallelization
- `audiocraft` - Meta's MusicGen model
- `qwen-vl-utils` - Utilities for Qwen models

Install dependencies:
```bash
cd ..
uv sync
```

## Usage

### Basic Usage

Run generation with default settings (medium mode, MusicGen-melody):

```bash
cd task2
uv run python run_generation.py
```

### Generation Modes

According to CLAUDE.md, there are three generation modes:

1. **Simple**: Text condition only
   ```bash
   uv run python run_generation.py --mode simple
   ```

2. **Medium**: Text + extracted features (melody, rhythm)
   ```bash
   uv run python run_generation.py --mode medium
   ```

3. **Strong**: Text + features + high classifier-free guidance
   ```bash
   uv run python run_generation.py --mode strong --cfg-scale 5.0
   ```

### Advanced Options

```bash
uv run python run_generation.py \
    --target-dir ../data/target_music_list_60s \
    --mode medium \
    --musicgen-model facebook/musicgen-melody \
    --captioner qwen \
    --duration 30.0 \
    --cfg-scale 3.0 \
    --temperature 1.0 \
    --output-dir results \
    --device cuda
```

### Arguments

- `--target-dir`: Directory with target music files
- `--mode`: Generation mode (simple/medium/strong)
- `--musicgen-model`: MusicGen model variant
  - `facebook/musicgen-small` (300M params)
  - `facebook/musicgen-medium` (1.5B params, default)
  - `facebook/musicgen-large` (3.3B params)
  - `facebook/musicgen-melody` (1.5B params with melody conditioning)
- `--captioner`: Audio captioning model (qwen)
- `--duration`: Generated music duration in seconds (default: 30.0)
- `--cfg-scale`: Classifier-free guidance scale (default: 3.0 for simple/medium, 5.0 for strong)
- `--temperature`: Sampling temperature (default: 1.0)
- `--output-dir`: Directory to save results (default: `results`)
- `--cache-dir`: Directory for caching captions (default: `cache`)
- `--device`: Device to use (cuda/cpu, default: cuda)
- `--skip-captioning`: Skip audio captioning and use cached captions

## Output

The script generates:

### 1. Generated Audio Files (`results/generated_audio/`)

Generated music files for each target track:
- Format: `generated_<target_name>.wav`
- Sample rate: 32kHz (MusicGen default)
- Duration: Specified by `--duration` parameter

### 2. Evaluation Results (`results/evaluation_results_<mode>.json`)

Detailed metrics for each generated track:

```json
{
  "target_song.wav": {
    "generated_file": "results/generated_audio/generated_target_song.wav",
    "caption": "An upbeat electronic dance track with synthesizers...",
    "mode": "medium",
    "cfg_scale": 3.0,
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

### 3. Cached Captions (`cache/captions_qwen.json`)

Audio captions for target tracks (reused across runs):

```json
{
  "target_song.wav": {
    "audio_path": "/path/to/target_song.wav",
    "file_name": "target_song.wav",
    "main_caption": "An upbeat electronic dance track...",
    "genre": "Electronic Dance Music (EDM)",
    "instruments": "Synthesizers, drum machine, bass",
    "mood": "Energetic and uplifting",
    "tempo": "Fast tempo around 128 BPM"
  }
}
```

## Evaluation Metrics

Same metrics as Task 1:

### 1. CLAP Similarity
- Cosine similarity between CLAP embeddings
- Range: [-1, 1], higher is more similar
- Measures overall audio similarity between generated and target

### 2. Melody Accuracy
- Frame-by-frame chromagram matching
- Range: [0, 1], higher is better
- Compares pitch class distributions

### 3. Audiobox Aesthetics
Four quality dimensions:
- **CE (Content Enjoyment)**: How enjoyable the content is
- **CU (Content Usefulness)**: How useful/appropriate the content is
- **PC (Production Complexity)**: Complexity of the production
- **PQ (Production Quality)**: Technical quality of the production

## Models

### Audio Captioning

#### Qwen-Audio (Default)
- Model: `Qwen/Qwen2-Audio-7B-Instruct`
- Multimodal large language model with audio understanding
- Generates detailed text descriptions of music
- **Status**: ✅ Recommended for high-quality captions

### Music Generation

#### MusicGen (Meta)
- **musicgen-small**: 300M params, fastest
- **musicgen-medium**: 1.5B params, balanced (recommended)
- **musicgen-large**: 3.3B params, highest quality
- **musicgen-melody**: 1.5B params with melody conditioning (recommended for medium/strong modes)

All models:
- Sample rate: 32kHz
- Maximum duration: 30 seconds
- Support text conditioning
- `musicgen-melody` additionally supports melody conditioning

### Feature Extraction

#### Melody Extraction
- Uses librosa for pitch detection and harmonic-percussive separation
- Extracts harmonic component as melody guide for MusicGen

#### Rhythm Extraction
- Tempo detection in BPM
- Beat tracking
- Onset strength envelope

#### Chord Extraction
- Chroma-based chord recognition
- Supports major and minor chords
- (Currently not used by MusicGen but available for future models)

## Important Rules (from CLAUDE.md)

1. ✅ Must use ALMs for music captioning
2. ❌ Cannot directly use target audio as condition
   - ❌ Cannot use auto-encoder to encode/decode target audio
   - ❌ Cannot use "audio condition" in MuseControlLite
   - ✅ Can only use features extracted with MIR tools (melody, rhythm, chords)
3. ✅ Can use different methods for each target track

## Example Workflow

```bash
# Step 1: Generate captions for all target tracks
uv run python run_generation.py --mode simple

# Step 2: Use cached captions for faster generation with melody
uv run python run_generation.py --mode medium --skip-captioning

# Step 3: Try strong mode with high CFG
uv run python run_generation.py --mode strong --cfg-scale 5.0 --skip-captioning

# Step 4: Compare results
cat results/evaluation_results_simple.json
cat results/evaluation_results_medium.json
cat results/evaluation_results_strong.json
```

## Performance Tips

1. **GPU Memory**: MusicGen-medium requires ~8GB VRAM, use `musicgen-small` for lower memory
2. **Caching**: Use `--skip-captioning` to reuse captions and speed up iterations
3. **Duration**: Shorter durations (e.g., 10-15s) generate faster for testing
4. **Batch Processing**: The script processes all target files automatically

## Troubleshooting

### Out of Memory
- Use smaller model: `--musicgen-model facebook/musicgen-small`
- Reduce duration: `--duration 15.0`
- Use CPU: `--device cpu` (slower but no memory limit)

### Import Errors
- Ensure all dependencies are installed: `cd .. && uv sync`
- Check task1 metrics are available: `ls ../task1/metrics/`

### Poor Quality Results
- Try strong mode: `--mode strong --cfg-scale 5.0`
- Use larger model: `--musicgen-model facebook/musicgen-large`
- Adjust temperature: `--temperature 0.8` (less random)

## Notes

1. Generated audio duration may be shorter than target (e.g., MusicGen max 30s, MuseControlLite max 47s)
2. Melody conditioning requires `musicgen-melody` model
3. First run downloads models and may take time
4. Captions are cached to speed up subsequent runs
