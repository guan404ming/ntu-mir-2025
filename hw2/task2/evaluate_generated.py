"""Simple script to evaluate already generated audio files."""

import json
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from task1.encoders.clap_encoder import CLAPEncoder
from task2.metrics import CLAPMetric, MelodyMetric, AestheticsMetric


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
    # Paths
    target_dir = Path("../data/target_music_list_60s")
    generated_dir = Path("results/generated_audio")
    output_file = Path("results/evaluation_results_simple.json")

    # Get target and generated files
    target_files = sorted(target_dir.glob("*.wav")) + sorted(target_dir.glob("*.mp3"))

    print("Initializing evaluation metrics...")
    clap_encoder = CLAPEncoder(device="cuda")
    clap_metric = CLAPMetric(clap_encoder)
    melody_metric = MelodyMetric()
    aesthetics_metric = AestheticsMetric(device="cuda")

    print("\nEvaluating generated music...")
    evaluation_results = {}

    for target_file in target_files:
        target_name = target_file.name
        generated_file = generated_dir / f"generated_{target_file.stem}.wav"

        if not generated_file.exists():
            print(f"Skipping {target_name}: Generated file not found")
            continue

        print(f"\nEvaluating: {target_name}")

        # CLAP similarity
        print("  - Calculating CLAP similarity...")
        clap_result = clap_metric.evaluate_retrieval(
            target_path=str(target_file),
            retrieved_path=str(generated_file),
        )

        # Melody accuracy
        print("  - Calculating melody accuracy...")
        melody_result = melody_metric.evaluate(
            target_path=str(target_file),
            generated_path=str(generated_file),
            target_duration=30.0,
        )

        # Aesthetics
        print("  - Calculating aesthetics metrics...")
        aesthetics_result = aesthetics_metric.evaluate_audio(str(generated_file))

        # Compile results
        evaluation_results[target_name] = {
            "generated_file": str(generated_file),
            "mode": "simple",
            "cfg_scale": 3.0,
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
        print(f"  CLAP similarity: {clap_result['clap_similarity']:.4f}")
        print(f"  Melody accuracy: {melody_result['melody_accuracy']:.4f}")
        print(f"  Aesthetics: CE={aesthetics_result['ce']:.3f}, CU={aesthetics_result['cu']:.3f}, PC={aesthetics_result['pc']:.3f}, PQ={aesthetics_result['pq']:.3f}")

    # Save results
    evaluation_results = convert_to_serializable(evaluation_results)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 80}")
    print("✓ Evaluation completed successfully!")
    print(f"Results saved to: {output_file}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
