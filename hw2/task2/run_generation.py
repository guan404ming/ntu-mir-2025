# ruff: noqa: E402
"""Main script for Task 2: Music Generation and Evaluation.

Task 2 Overview (from CLAUDE.md):
- Goal: Generate music similar to target tracks using text-to-music models
- Method:
  1. Audio Captioning: Use ALMs to describe the target music
  2. Feature Extraction: Extract MIR features (melody, rhythm) from target
  3. Text-to-Music Generation: Generate music using text + extracted features

Evaluation Metrics (same as Task 1):
1. CLAP similarity: Cosine similarity between generated and target music
2. Aesthetics metrics: Quality metrics (CE, CU, PC, PQ) of generated music
3. Melody accuracy: Melody similarity between generated and target music

Generation Modes:
- Simple: Text condition only
- Medium: Text + extracted features (melody/rhythm)
- Strong: Text + features + adjusted CFG scale
"""

import argparse
import json
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# IMPORTANT: Save sys.argv before importing any modules that use laion_clap
# laion_clap has module-level argparse that interferes with our script's arguments
_saved_argv = sys.argv.copy()
sys.argv = [sys.argv[0]]  # Keep only the script name

from task2.captioning.qwen_captioner import QwenAudioCaptioner
from task2.generation.musicgen_generator import MusicGenGenerator
from task2.features.melody_extractor import MelodyExtractor
from task2.features.rhythm_extractor import RhythmExtractor
from task2.metrics import CLAPMetric, MelodyMetric, AestheticsMetric
from task1.encoders.clap_encoder import CLAPEncoder

# Restore sys.argv after imports
sys.argv = _saved_argv


def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    import numpy as np

    if isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(v) for v in obj]
    return obj


def main():
    parser = argparse.ArgumentParser(
        description="Task 2: Music Generation and Evaluation"
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        default="../data/target_music_list_60s",
        help="Directory containing target music files",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="medium",
        choices=["simple", "medium", "strong"],
        help="Generation mode (simple=text only, medium=text+features, strong=text+features+high CFG)",
    )
    parser.add_argument(
        "--musicgen-model",
        type=str,
        default="facebook/musicgen-melody",
        choices=[
            "facebook/musicgen-small",
            "facebook/musicgen-medium",
            "facebook/musicgen-large",
            "facebook/musicgen-melody",
        ],
        help="MusicGen model to use",
    )
    parser.add_argument(
        "--captioner",
        type=str,
        default="qwen",
        choices=["qwen"],
        help="Audio captioning model to use",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Duration of generated music (seconds)",
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=None,
        help="CFG scale for generation (default: 3.0 for simple/medium, 5.0 for strong)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="cache",
        help="Directory for caching captions",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda/cpu)",
    )
    parser.add_argument(
        "--skip-captioning",
        action="store_true",
        help="Skip audio captioning (use cached captions)",
    )

    args = parser.parse_args()

    # Set default CFG scale based on mode
    if args.cfg_scale is None:
        args.cfg_scale = 5.0 if args.mode == "strong" else 3.0

    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    generated_dir = output_dir / "generated_audio"
    generated_dir.mkdir(exist_ok=True)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print("Task 2: Music Generation and Evaluation")
    print("=" * 80)
    print(f"Mode: {args.mode}")
    print(f"MusicGen model: {args.musicgen_model}")
    print(f"Captioner: {args.captioner}")
    print(f"Duration: {args.duration}s")
    print(f"CFG scale: {args.cfg_scale}")
    print(f"Temperature: {args.temperature}")
    print(f"Target directory: {args.target_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {args.device}")
    print("=" * 80)

    # Get target files
    target_dir = Path(args.target_dir)
    audio_extensions = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}
    target_files = sorted(
        [f for f in target_dir.iterdir() if f.suffix.lower() in audio_extensions]
    )

    print(f"\nFound {len(target_files)} target files")

    # Initialize models
    print("\n[1/6] Initializing models...")

    # Audio captioning (loaded temporarily only if needed)
    captioner = None

    # Music generation - will be loaded after captioning
    generator = None

    # Feature extractors (for medium and strong modes) - will be loaded later
    melody_extractor = None
    rhythm_extractor = None

    # Evaluation metrics - will be loaded later
    clap_metric = None
    melody_metric = None
    aesthetics_metric = None

    # Process each target file
    print("\n[2/6] Audio Captioning...")
    captions = {}
    caption_cache_file = cache_dir / f"captions_{args.captioner}.json"

    if args.skip_captioning and caption_cache_file.exists():
        print(f"  Loading cached captions from {caption_cache_file}")
        with open(caption_cache_file, "r", encoding="utf-8") as f:
            captions = json.load(f)
    else:
        # Load captioner only when needed
        print("  Loading audio captioner...")
        if args.captioner == "qwen":
            # Use 8-bit quantization to reduce memory usage (~50% memory savings)
            captioner = QwenAudioCaptioner(device=args.device, use_8bit=True)

        for i, target_file in enumerate(target_files):
            print(f"\n  [{i + 1}/{len(target_files)}] Captioning: {target_file.name}")
            caption_result = captioner.generate_detailed_caption(str(target_file))
            captions[target_file.name] = caption_result
            print(f"  Caption: {caption_result['main_caption']}")

        # Save captions
        with open(caption_cache_file, "w", encoding="utf-8") as f:
            json.dump(captions, f, indent=2, ensure_ascii=False)
        print(f"\n  Saved captions to: {caption_cache_file}")

        # Free up GPU memory by deleting the captioner
        print("  Unloading audio captioner to free GPU memory...")
        del captioner
        import gc
        import torch

        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()

    # Generate music
    print("\n[3/6] Feature Extraction...")
    features = {}
    if args.mode in ["medium", "strong"]:
        # Initialize feature extractors
        print("  Initializing feature extractors...")
        # Use a temporary generator sample rate (32kHz for MusicGen)
        melody_extractor = MelodyExtractor(sr=32000)
        rhythm_extractor = RhythmExtractor(sr=32000)

        for i, target_file in enumerate(target_files):
            print(
                f"\n  [{i + 1}/{len(target_files)}] Extracting features: {target_file.name}"
            )

            # Extract melody
            melody = melody_extractor.extract_melody_for_musicgen(
                str(target_file), duration=args.duration
            )

            # Extract rhythm
            rhythm = rhythm_extractor.extract_rhythm_features(str(target_file))

            features[target_file.name] = {
                "melody": melody,
                "tempo": rhythm["tempo"],
            }
            print(f"  Tempo: {rhythm['tempo']:.1f} BPM")

    # Load music generator after captioning (to free GPU memory)
    print("\n  Loading music generator...")
    generator = MusicGenGenerator(model_name=args.musicgen_model, device=args.device)

    # Generate music for each target
    print("\n[4/6] Music Generation...")
    generation_results = {}

    for i, target_file in enumerate(target_files):
        print(f"\n{'=' * 60}")
        print(f"[{i + 1}/{len(target_files)}] Generating: {target_file.name}")
        print(f"{'=' * 60}")

        # Get caption
        caption = captions[target_file.name]["main_caption"]
        print(f"Caption: {caption}")

        # Generate based on mode
        output_file = generated_dir / f"generated_{target_file.stem}.wav"

        if args.mode == "simple":
            # Simple: Text only
            print("Mode: Simple (text only)")
            audio = generator.generate(
                prompt=caption,
                duration=args.duration,
                guidance_scale=args.cfg_scale,
                temperature=args.temperature,
            )

        elif args.mode in ["medium", "strong"]:
            # Medium/Strong: Text + melody
            print(f"Mode: {args.mode.capitalize()} (text + melody)")
            melody = features[target_file.name]["melody"]
            tempo = features[target_file.name]["tempo"]

            # Enhance prompt with tempo info
            enhanced_prompt = f"{caption} The tempo is around {tempo:.0f} BPM."
            print(f"Enhanced prompt: {enhanced_prompt}")

            # Check if model supports melody
            if generator.is_melody_model:
                audio = generator.generate_with_melody(
                    prompt=enhanced_prompt,
                    melody=melody,
                    melody_sr=generator.sample_rate,
                    duration=args.duration,
                    guidance_scale=args.cfg_scale,
                    temperature=args.temperature,
                )
            else:
                print(
                    "Warning: Model doesn't support melody conditioning, using text only"
                )
                audio = generator.generate(
                    prompt=enhanced_prompt,
                    duration=args.duration,
                    guidance_scale=args.cfg_scale,
                    temperature=args.temperature,
                )

        # Save generated audio
        generator.save_audio(audio, str(output_file))

        generation_results[target_file.name] = {
            "generated_file": str(output_file),
            "caption": caption,
            "mode": args.mode,
            "cfg_scale": args.cfg_scale,
        }

    # Evaluate generated music
    print("\n[5/6] Evaluating Generated Music...")

    # Load evaluation metrics
    print("  Loading evaluation metrics...")
    clap_encoder = CLAPEncoder(device=args.device)
    clap_metric = CLAPMetric(clap_encoder)
    melody_metric = MelodyMetric()
    aesthetics_metric = AestheticsMetric(device=args.device)

    evaluation_results = {}

    for target_file in target_files:
        target_name = target_file.name
        print(f"\n{'=' * 60}")
        print(f"Evaluating: {target_name}")
        print(f"{'=' * 60}")

        generated_path = generation_results[target_name]["generated_file"]

        # 1. CLAP similarity
        print("  - Calculating CLAP similarity...")
        clap_result = clap_metric.evaluate_retrieval(
            target_path=str(target_file),
            retrieved_path=generated_path,
        )

        # 2. Melody accuracy
        print("  - Calculating melody accuracy...")
        melody_result = melody_metric.evaluate(
            target_path=str(target_file),
            generated_path=generated_path,
            target_duration=args.duration,
        )

        # 3. Aesthetics
        print("  - Calculating aesthetics metrics...")
        aesthetics_result = aesthetics_metric.evaluate_audio(generated_path)

        # Compile results
        evaluation_results[target_name] = {
            "generated_file": generation_results[target_name]["generated_file"],
            "caption": generation_results[target_name]["caption"],
            "mode": args.mode,
            "cfg_scale": args.cfg_scale,
            "clap_similarity": clap_result["clap_similarity"],
            "melody_accuracy": melody_result["melody_accuracy"],
            "aesthetics": {
                "ce": aesthetics_result["ce"],
                "cu": aesthetics_result["cu"],
                "pc": aesthetics_result["pc"],
                "pq": aesthetics_result["pq"],
            },
        }

        print(f"\nResults for {target_name}:")
        print(f"  1. CLAP similarity: {clap_result['clap_similarity']:.4f}")
        print(f"  2. Melody accuracy: {melody_result['melody_accuracy']:.4f}")
        print("  3. Aesthetics:")
        print(f"     - CE (Content Enjoyment): {aesthetics_result['ce']:.3f}")
        print(f"     - CU (Content Usefulness): {aesthetics_result['cu']:.3f}")
        print(f"     - PC (Production Complexity): {aesthetics_result['pc']:.3f}")
        print(f"     - PQ (Production Quality): {aesthetics_result['pq']:.3f}")

    # Save results
    print("\n[6/6] Saving Results...")
    results_file = output_dir / f"evaluation_results_{args.mode}.json"
    evaluation_results = convert_to_serializable(evaluation_results)
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 80}")
    print("✓ Task 2 completed successfully!")
    print(f"Generated audio saved to: {generated_dir}")
    print(f"Evaluation results saved to: {results_file}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
