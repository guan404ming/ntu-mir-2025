"""Evaluate melody accuracy for generated audio."""

import json
from pathlib import Path
import sys
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from task2.metrics import MelodyMetric


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
    output_file = Path("results/melody_evaluation_results_simple.json")

    # Get target and generated files
    target_files = sorted(target_dir.glob("*.wav")) + sorted(target_dir.glob("*.mp3"))

    print("=" * 80)
    print("Task 2 Melody Evaluation - Simple Mode")
    print("=" * 80)
    print(f"\nFound {len(target_files)} target files")
    print(f"Generated audio directory: {generated_dir}")

    print("\nInitializing melody metric...")
    melody_metric = MelodyMetric()

    print("\n" + "=" * 80)
    print("Evaluating melody accuracy...")
    print("=" * 80)

    evaluation_results = {}

    for i, target_file in enumerate(target_files, 1):
        target_name = target_file.name
        generated_file = generated_dir / f"generated_{target_file.stem}.wav"

        if not generated_file.exists():
            print(f"\n[{i}/{len(target_files)}] Skipping {target_name}: Generated file not found")
            continue

        print(f"\n{'=' * 60}")
        print(f"[{i}/{len(target_files)}] {target_name}")
        print(f"{'=' * 60}")
        print(f"  Target:    {target_file.name}")
        print(f"  Generated: {generated_file.name}")

        # Melody accuracy
        print("  Calculating melody accuracy...")
        melody_result = melody_metric.evaluate(
            target_path=str(target_file),
            generated_path=str(generated_file),
            target_duration=30.0,
        )

        # Compile results
        evaluation_results[target_name] = {
            "generated_file": str(generated_file),
            "target_file": str(target_file),
            "mode": "simple",
            "melody_accuracy": melody_result["melody_accuracy"],
        }

        print(f"  Melody accuracy: {melody_result['melody_accuracy']:.4f}")

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

        print(f"\nMelody Accuracy:")
        print(f"  Average: {np.mean(melody_scores):.4f}")
        print(f"  Std Dev: {np.std(melody_scores):.4f}")
        print(f"  Min:     {np.min(melody_scores):.4f}")
        print(f"  Max:     {np.max(melody_scores):.4f}")

        print(f"\nIndividual Scores:")
        for name, result in evaluation_results.items():
            print(f"  {name:60s} {result['melody_accuracy']:.4f}")

    print("\n" + "=" * 80)
    print("✓ Melody evaluation completed successfully!")
    print(f"Results saved to: {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
