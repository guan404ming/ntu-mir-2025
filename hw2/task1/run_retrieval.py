"""Main script for Task 1: Music Retrieval and Evaluation."""

import argparse
import json
from pathlib import Path
import sys
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from task1.encoders.clap_encoder import CLAPEncoder
from task1.retrieval import MusicRetrieval
from task1.metrics.clap_metric import CLAPMetric
from task1.metrics.melody_metric import MelodyMetric
from task1.metrics.aesthetics_metric import AestheticsMetric


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
    parser = argparse.ArgumentParser(
        description="Task 1: Music Retrieval and Evaluation"
    )
    parser.add_argument(
        "--reference-dir",
        type=str,
        default="../data/referecne_music_list_60s",
        help="Directory containing reference music files",
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        default="../data/target_music_list_60s",
        help="Directory containing target music files",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="clap",
        choices=["clap", "music2latent", "muq"],
        help="Audio encoder to use for retrieval",
    )
    parser.add_argument(
        "--top-k", type=int, default=5, help="Number of similar tracks to retrieve"
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
        help="Directory for caching embeddings",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda/cpu)",
    )
    parser.add_argument(
        "--target-duration",
        type=float,
        default=None,
        help="Duration to trim target audio for melody accuracy (seconds)",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print("Task 1: Music Retrieval and Evaluation")
    print("=" * 80)
    print(f"Encoder: {args.encoder}")
    print(f"Reference directory: {args.reference_dir}")
    print(f"Target directory: {args.target_dir}")
    print(f"Top-K: {args.top_k}")
    print(f"Device: {args.device}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80)

    # Initialize encoder
    print("\n[1/5] Initializing audio encoder...")
    if args.encoder == "clap":
        encoder = CLAPEncoder(device=args.device)
    elif args.encoder == "music2latent":
        from task1.encoders.music2latent_encoder import Music2LatentEncoder

        encoder = Music2LatentEncoder(device=args.device)
    elif args.encoder == "muq":
        from task1.encoders.muq_encoder import MuQEncoder

        encoder = MuQEncoder(device=args.device)
    else:
        raise ValueError(f"Unknown encoder: {args.encoder}")

    # Initialize retrieval system
    print("\n[2/5] Setting up retrieval system...")
    retrieval = MusicRetrieval(
        encoder=encoder,
        reference_dir=args.reference_dir,
        cache_dir=args.cache_dir,
    )

    # Perform retrieval for all target files
    print("\n[3/5] Retrieving similar music for all targets...")
    retrieval_results = retrieval.retrieve_all_targets(
        target_dir=args.target_dir,
        top_k=args.top_k,
    )

    # Save retrieval results
    retrieval_output = output_dir / f"retrieval_results_{args.encoder}.json"
    retrieval.save_results(retrieval_results, str(retrieval_output))

    # Initialize evaluation metrics
    print("\n[4/5] Initializing evaluation metrics...")
    clap_metric = CLAPMetric(
        encoder if args.encoder == "clap" else CLAPEncoder(device=args.device)
    )
    melody_metric = MelodyMetric()
    aesthetics_metric = AestheticsMetric(device=args.device)

    # Evaluate each target with its top retrieved track
    print("\n[5/5] Evaluating retrieved music...")
    evaluation_results = {}

    for target_path, similar_tracks in retrieval_results.items():
        target_name = Path(target_path).name
        print(f"\n{'=' * 60}")
        print(f"Evaluating: {target_name}")
        print(f"{'=' * 60}")

        # Get top retrieved track
        top_retrieved_path, retrieval_score = similar_tracks[0]

        # 1. CLAP similarity
        print("  - Calculating CLAP similarity...")
        clap_result = clap_metric.evaluate_retrieval(
            target_path=target_path,
            retrieved_path=top_retrieved_path,
        )

        # 2. Melody accuracy
        print("  - Calculating melody accuracy...")
        melody_result = melody_metric.evaluate(
            target_path=target_path,
            generated_path=top_retrieved_path,
            target_duration=args.target_duration,
        )

        # 3. Aesthetics metrics
        print("  - Calculating aesthetics metrics...")
        aesthetics_result = aesthetics_metric.evaluate_audio(top_retrieved_path)

        # Compile results
        evaluation_results[target_name] = {
            "retrieved_track": Path(top_retrieved_path).name,
            "retrieval_similarity": float(retrieval_score),
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
        print(f"  Retrieved: {Path(top_retrieved_path).name}")
        print(f"  Retrieval similarity: {retrieval_score:.4f}")
        print(f"  CLAP similarity: {clap_result['clap_similarity']:.4f}")
        print(f"  Melody accuracy: {melody_result['melody_accuracy']:.4f}")
        print("  Aesthetics:")
        print(f"    CE: {aesthetics_result['ce']:.3f}")
        print(f"    CU: {aesthetics_result['cu']:.3f}")
        print(f"    PC: {aesthetics_result['pc']:.3f}")
        print(f"    PQ: {aesthetics_result['pq']:.3f}")

    # Save evaluation results
    evaluation_output = output_dir / f"evaluation_results_{args.encoder}.json"
    # Convert numpy types to native Python types
    evaluation_results = convert_to_serializable(evaluation_results)
    with open(evaluation_output, "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 80}")
    print("✓ Task 1 completed successfully!")
    print(f"Retrieval results saved to: {retrieval_output}")
    print(f"Evaluation results saved to: {evaluation_output}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
