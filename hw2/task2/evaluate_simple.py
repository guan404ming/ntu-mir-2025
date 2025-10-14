"""Simple evaluation script for generated audio - computes available metrics."""

import json
from pathlib import Path
import sys
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from task2.metrics import MelodyMetric, AestheticsMetric


def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
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
    # Paths
    target_dir = Path("../data/target_music_list_60s")
    generated_dir = Path("results/generated_audio")
    output_file = Path("results/evaluation_results_simple.json")

    # Get target and generated files
    target_files = sorted(target_dir.glob("*.wav")) + sorted(target_dir.glob("*.mp3"))

    print("=" * 80)
    print("Task 2 Evaluation - Simple Mode")
    print("=" * 80)
    print(f"\nFound {len(target_files)} target files")
    print(f"Generated audio directory: {generated_dir}")

    print("\nInitializing evaluation metrics...")
    melody_metric = MelodyMetric()
    aesthetics_metric = AestheticsMetric(device="cuda")

    print("\n" + "=" * 80)
    print("Evaluating generated music...")
    print("=" * 80)

    evaluation_results = {}

    for i, target_file in enumerate(target_files, 1):
        target_name = target_file.name
        generated_file = generated_dir / f"generated_{target_file.stem}.wav"

        if not generated_file.exists():
            print(f"\n[{i}/{len(target_files)}] Skipping {target_name}: Generated file not found")
            continue

        print(f"\n{'=' * 60}")
        print(f"[{i}/{len(target_files)}] Evaluating: {target_name}")
        print(f"{'=' * 60}")

        # Melody accuracy
        print("  1. Calculating melody accuracy...")
        melody_result = melody_metric.evaluate(
            target_path=str(target_file),
            generated_path=str(generated_file),
            target_duration=30.0,
        )

        # Aesthetics
        print("  2. Calculating aesthetics metrics...")
        aesthetics_result = aesthetics_metric.evaluate_audio(str(generated_file))

        # Compile results
        evaluation_results[target_name] = {
            "generated_file": str(generated_file),
            "target_file": str(target_file),
            "mode": "simple",
            "cfg_scale": 3.0,
            "melody_accuracy": melody_result["melody_accuracy"],
            "aesthetics": {
                "ce": aesthetics_result["ce"],
                "cu": aesthetics_result["cu"],
                "pc": aesthetics_result["pc"],
                "pq": aesthetics_result["pq"],
            },
        }

        print(f"\n  Results for {target_name}:")
        print(f"    Melody accuracy: {melody_result['melody_accuracy']:.4f}")
        print(f"    Aesthetics:")
        print(f"      - CE (Content Enjoyment):     {aesthetics_result['ce']:.3f}")
        print(f"      - CU (Content Usefulness):    {aesthetics_result['cu']:.3f}")
        print(f"      - PC (Production Complexity): {aesthetics_result['pc']:.3f}")
        print(f"      - PQ (Production Quality):    {aesthetics_result['pq']:.3f}")

    # Save results
    evaluation_results = convert_to_serializable(evaluation_results)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)

    if evaluation_results:
        melody_scores = [r["melody_accuracy"] for r in evaluation_results.values()]
        ce_scores = [r["aesthetics"]["ce"] for r in evaluation_results.values()]
        cu_scores = [r["aesthetics"]["cu"] for r in evaluation_results.values()]
        pc_scores = [r["aesthetics"]["pc"] for r in evaluation_results.values()]
        pq_scores = [r["aesthetics"]["pq"] for r in evaluation_results.values()]

        print(f"\nMelody Accuracy:")
        print(f"  Average: {np.mean(melody_scores):.4f}")
        print(f"  Std Dev: {np.std(melody_scores):.4f}")
        print(f"  Min:     {np.min(melody_scores):.4f}")
        print(f"  Max:     {np.max(melody_scores):.4f}")

        print(f"\nAesthetics - Content Enjoyment (CE):")
        print(f"  Average: {np.mean(ce_scores):.3f}")
        print(f"  Std Dev: {np.std(ce_scores):.3f}")

        print(f"\nAesthetics - Content Usefulness (CU):")
        print(f"  Average: {np.mean(cu_scores):.3f}")
        print(f"  Std Dev: {np.std(cu_scores):.3f}")

        print(f"\nAesthetics - Production Complexity (PC):")
        print(f"  Average: {np.mean(pc_scores):.3f}")
        print(f"  Std Dev: {np.std(pc_scores):.3f}")

        print(f"\nAesthetics - Production Quality (PQ):")
        print(f"  Average: {np.mean(pq_scores):.3f}")
        print(f"  Std Dev: {np.std(pq_scores):.3f}")

    print("\n" + "=" * 80)
    print("✓ Evaluation completed successfully!")
    print(f"Results saved to: {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
