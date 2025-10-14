# Quick Start Guide for Task 2

## Installation

All dependencies have been installed via `uv sync`. The following packages were added:
- `transformers>=4.50.0` - For Qwen-Audio model
- `accelerate>=0.20.0` - For model parallelization
- `audiocraft>=1.3.0` - For MusicGen model
- `qwen-vl-utils>=0.0.8` - For Qwen model utilities

## Running Task 2

### Step 1: Basic Text-Only Generation (Simple Mode)

Generate music using only text captions:

```bash
cd /home/gmchiu/Documents/Github/ntu-mir-2025/hw2/task2
uv run python run_generation.py --mode simple --duration 30.0
```

### Step 2: Text + Melody Generation (Medium Mode)

Generate music using text captions + extracted melody features:

```bash
uv run python run_generation.py --mode medium --duration 30.0 --skip-captioning
```

Note: `--skip-captioning` reuses captions from Step 1 to save time.

### Step 3: High-CFG Generation (Strong Mode)

Generate with higher classifier-free guidance for better prompt adherence:

```bash
uv run python run_generation.py --mode strong --cfg-scale 5.0 --duration 30.0 --skip-captioning
```

## Generation Modes Explained

### Simple Mode
- **Method**: Text description only
- **CFG Scale**: 3.0 (default)
- **Use case**: Quick generation, baseline

### Medium Mode
- **Method**: Text + Melody conditioning
- **Features**: Extracts melody (harmonic content) and tempo from target
- **CFG Scale**: 3.0 (default)
- **Use case**: Better alignment with target musical characteristics

### Strong Mode
- **Method**: Text + Melody + High CFG
- **Features**: Same as medium + increased guidance scale
- **CFG Scale**: 5.0 (default, adjustable)
- **Use case**: Strongest adherence to prompt and features

## Key Parameters

### Model Selection
- `--musicgen-model facebook/musicgen-small` - Fast, 300M params
- `--musicgen-model facebook/musicgen-medium` - Balanced, 1.5B params (default)
- `--musicgen-model facebook/musicgen-large` - Best quality, 3.3B params
- `--musicgen-model facebook/musicgen-melody` - With melody conditioning (recommended for medium/strong)

### Generation Parameters
- `--duration 30.0` - Duration in seconds (max 30s for MusicGen)
- `--cfg-scale 3.0` - Classifier-free guidance (higher = more prompt adherence)
- `--temperature 1.0` - Sampling temperature (higher = more random)

### Device
- `--device cuda` - Use GPU (default, recommended)
- `--device cpu` - Use CPU (slower)

## Output Files

After running, you'll find:

1. **Generated Audio**: `results/generated_audio/generated_<target>.wav`
2. **Evaluation Results**: `results/evaluation_results_<mode>.json`
3. **Cached Captions**: `cache/captions_qwen.json`

## Example Workflows

### Quick Test (1 file, fast)
```bash
# Test with small model and short duration
uv run python run_generation.py \
    --mode simple \
    --musicgen-model facebook/musicgen-small \
    --duration 10.0
```

### Full Generation Pipeline
```bash
# 1. Generate captions and simple mode
uv run python run_generation.py --mode simple --duration 30.0

# 2. Medium mode with cached captions
uv run python run_generation.py --mode medium --skip-captioning

# 3. Strong mode with high CFG
uv run python run_generation.py --mode strong --cfg-scale 5.0 --skip-captioning

# 4. Compare results
cat results/evaluation_results_simple.json
cat results/evaluation_results_medium.json
cat results/evaluation_results_strong.json
```

### Production Quality
```bash
# Use best model with melody conditioning
uv run python run_generation.py \
    --mode strong \
    --musicgen-model facebook/musicgen-melody \
    --cfg-scale 5.0 \
    --temperature 0.8 \
    --duration 30.0 \
    --skip-captioning
```

## Evaluation Metrics

For each generated track, you get:

1. **CLAP Similarity** (0-1): Audio similarity to target
2. **Melody Accuracy** (0-1): Pitch class similarity
3. **Aesthetics** (0-1 each):
   - CE: Content Enjoyment
   - CU: Content Usefulness
   - PC: Production Complexity
   - PQ: Production Quality

## Tips for Better Results

1. **First run will be slow** - Models need to download (~6GB for MusicGen-medium)
2. **Use caching** - Always use `--skip-captioning` after first run
3. **GPU is important** - CPU generation is very slow
4. **Try different CFG scales** - Range 2.0-7.0, higher = more adherence
5. **Melody model for medium/strong** - Use `musicgen-melody` for better conditioning

## Troubleshooting

### Out of Memory
```bash
# Use smaller model
uv run python run_generation.py --musicgen-model facebook/musicgen-small

# Or shorter duration
uv run python run_generation.py --duration 15.0

# Or CPU (slow)
uv run python run_generation.py --device cpu
```

### Import Errors
```bash
# Re-sync dependencies
cd /home/gmchiu/Documents/Github/ntu-mir-2025/hw2
uv sync
```

### Poor Quality
```bash
# Try strong mode with high CFG
uv run python run_generation.py \
    --mode strong \
    --cfg-scale 6.0 \
    --temperature 0.7 \
    --musicgen-model facebook/musicgen-large
```

## Project Structure

```
task2/
├── captioning/              # Audio-to-text models
│   ├── base_captioner.py
│   └── qwen_captioner.py    # Qwen-Audio implementation
├── generation/              # Text-to-music models
│   ├── base_generator.py
│   └── musicgen_generator.py # MusicGen implementation
├── features/                # MIR feature extraction
│   ├── melody_extractor.py  # Melody/pitch extraction
│   ├── rhythm_extractor.py  # Tempo/beat extraction
│   └── chord_extractor.py   # Chord recognition
├── metrics/                 # Evaluation (from task1)
│   └── __init__.py
├── cache/                   # Cached captions
├── results/                 # Output files
│   └── generated_audio/     # Generated wav files
├── run_generation.py        # Main script
└── README.md               # Full documentation
```

## Next Steps

1. Run simple mode to generate captions and baseline
2. Compare simple vs medium vs strong modes
3. Try different models and CFG scales
4. Analyze evaluation results to pick best approach
5. Generate final submissions with best configuration
