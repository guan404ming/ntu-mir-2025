#!/usr/bin/env python3
"""
Standalone script to check test predictions against ground truth answers.
This script uses the same logic as count_score.py but with more detailed output.
"""

import json
import sys
import argparse


def check_accuracy(answers_file, predictions_file):
    """Check accuracy of predictions against ground truth answers"""

    try:
        # Load ground truth answers
        with open(answers_file, "r") as f:
            answers = json.load(f)

        # Load predictions
        with open(predictions_file, "r") as f:
            predictions = json.load(f)

        print(f"Loaded {len(answers)} ground truth answers")
        print(f"Loaded predictions for {len(predictions)} test files")

        top1_correct = 0
        top3_correct = 0
        total = len(answers)

        # Track per-class performance
        class_stats = {}
        incorrect_predictions = []

        for i, answer in enumerate(answers):
            file_id = f"{i+1:03d}"  # Convert to 001, 002, etc.

            if file_id not in predictions:
                print(f"Warning: No prediction found for file {file_id}")
                continue

            pred = predictions[file_id]

            # Initialize class stats if not seen before
            if answer not in class_stats:
                class_stats[answer] = {"total": 0, "top1_correct": 0, "top3_correct": 0}
            class_stats[answer]["total"] += 1

            # Check top-1 accuracy
            if answer == pred[0]:
                top1_correct += 1
                top3_correct += 1
                class_stats[answer]["top1_correct"] += 1
                class_stats[answer]["top3_correct"] += 1
            # Check top-3 accuracy
            elif answer in pred[:3]:
                top3_correct += 1
                class_stats[answer]["top3_correct"] += 1
            else:
                # Track incorrect predictions for analysis
                incorrect_predictions.append({
                    "file_id": file_id,
                    "ground_truth": answer,
                    "predictions": pred
                })

        # Calculate overall accuracy
        top1_accuracy = top1_correct / total
        top3_accuracy = top3_correct / total

        print("\n" + "="*60)
        print("OVERALL TEST SET PERFORMANCE")
        print("="*60)
        print(f"Top-1 Accuracy: {top1_accuracy:.4f} ({top1_correct}/{total})")
        print(f"Top-3 Accuracy: {top3_accuracy:.4f} ({top3_correct}/{total})")

        # Per-class performance
        print("\n" + "="*60)
        print("PER-CLASS PERFORMANCE")
        print("="*60)
        print(f"{'Artist':<25} {'Total':<6} {'Top-1':<12} {'Top-3':<12}")
        print("-" * 60)

        for artist in sorted(class_stats.keys()):
            stats = class_stats[artist]
            top1_acc = stats["top1_correct"] / stats["total"] if stats["total"] > 0 else 0
            top3_acc = stats["top3_correct"] / stats["total"] if stats["total"] > 0 else 0
            print(f"{artist:<25} {stats['total']:<6} {top1_acc:<12.3f} {top3_acc:<12.3f}")

        # Show some incorrect predictions for analysis
        if incorrect_predictions and len(incorrect_predictions) <= 10:
            print("\n" + "="*60)
            print("INCORRECT PREDICTIONS (All shown)")
            print("="*60)
            for item in incorrect_predictions:
                print(f"File {item['file_id']}: {item['ground_truth']} -> {item['predictions']}")
        elif incorrect_predictions:
            print("\n" + "="*60)
            print("INCORRECT PREDICTIONS (First 10 shown)")
            print("="*60)
            for item in incorrect_predictions[:10]:
                print(f"File {item['file_id']}: {item['ground_truth']} -> {item['predictions']}")
            print(f"... and {len(incorrect_predictions)-10} more")

        # Save detailed results
        results = {
            "overall": {
                "top1_accuracy": top1_accuracy,
                "top3_accuracy": top3_accuracy,
                "top1_correct": top1_correct,
                "top3_correct": top3_correct,
                "total_samples": total
            },
            "per_class": class_stats,
            "incorrect_predictions": incorrect_predictions
        }

        output_file = predictions_file.replace(".json", "_accuracy_check.json")
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nDetailed results saved to: {output_file}")
        return top1_accuracy, top3_accuracy

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        return None, None
    except Exception as e:
        print(f"Error checking accuracy: {e}")
        return None, None


def main():
    parser = argparse.ArgumentParser(description="Check test predictions against ground truth")
    parser.add_argument("answers_file", help="Path to ground truth answers JSON file")
    parser.add_argument("predictions_file", help="Path to predictions JSON file")

    args = parser.parse_args()

    print(f"Checking predictions in {args.predictions_file}")
    print(f"Against ground truth in {args.answers_file}")

    top1_acc, top3_acc = check_accuracy(args.answers_file, args.predictions_file)

    if top1_acc is not None:
        print(f"\nFinal Result: Top-1={top1_acc:.4f}, Top-3={top3_acc:.4f}")
    else:
        print("Failed to compute accuracy")
        sys.exit(1)


if __name__ == "__main__":
    main()