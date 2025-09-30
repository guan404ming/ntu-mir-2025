"""
Task 1 Report Generation Script
Generates all visualizations for Task 1 (Traditional ML) report
"""
import numpy as np
import json
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os

# Set style
plt.style.use('default')
sns.set_palette("husl")


def load_results():
    """Load trained models and evaluation results"""
    print("Loading Task 1 results...")

    # Load results summary
    with open("results/task1/results_summary.json", "r") as f:
        results_summary = json.load(f)

    # Load label encoder
    with open("results/task1/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    # Load validation data
    y_val = np.load("results/task1/y_val.npy")

    # Encode labels
    y_val_encoded = label_encoder.transform(y_val)

    print(f"Best model: {results_summary['best_model']}")
    print(f"Number of classes: {len(label_encoder.classes_)}")

    return results_summary, label_encoder, y_val_encoded


def generate_confusion_matrix(best_model, y_val, label_encoder, output_dir):
    """Generate confusion matrix for best model"""
    print("\nGenerating confusion matrix...")

    # Load best model
    with open("results/task1/best_model.pkl", "rb") as f:
        model = pickle.load(f)

    # Load scaled validation data
    X_val_scaled = np.load("results/task1/X_val_scaled.npy")

    # Make predictions
    y_pred = model.predict(X_val_scaled)

    # Generate confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    class_names = label_encoder.classes_

    plt.figure(figsize=(16, 12))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(f"Confusion Matrix - {best_model}", fontsize=16, fontweight='bold')
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("Actual", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"task1_confusion_matrix_{best_model.lower()}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(f"Confusion matrix saved to {output_dir}/task1_confusion_matrix_{best_model.lower()}.png")

    # Print classification report
    print(f"\nClassification Report for {best_model}:")
    print(classification_report(y_val, y_pred, target_names=class_names))


def generate_model_comparison(results_summary, output_dir):
    """Generate model comparison charts"""
    print("\nGenerating model comparison chart...")

    model_performance = results_summary['model_performance']
    model_names = list(model_performance.keys())
    top1_scores = [model_performance[name]["top1_accuracy"] for name in model_names]
    top3_scores = [model_performance[name]["top3_accuracy"] for name in model_names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Top-1 accuracy comparison
    bars1 = ax1.bar(
        model_names, top1_scores, color=["#ff7f0e", "#2ca02c", "#1f77b4"]
    )
    ax1.set_title("Top-1 Accuracy Comparison", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3, axis='y')
    for bar, score in zip(bars1, top1_scores):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{score:.3f}",
            ha="center",
            va="bottom",
            fontweight='bold'
        )

    # Top-3 accuracy comparison
    bars2 = ax2.bar(
        model_names, top3_scores, color=["#ff7f0e", "#2ca02c", "#1f77b4"]
    )
    ax2.set_title("Top-3 Accuracy Comparison", fontsize=14, fontweight='bold')
    ax2.set_ylabel("Accuracy", fontsize=12)
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3, axis='y')
    for bar, score in zip(bars2, top3_scores):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{score:.3f}",
            ha="center",
            va="bottom",
            fontweight='bold'
        )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "task1_model_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Model comparison chart saved to {output_dir}/task1_model_comparison.png")


def main():
    """Main function to generate all Task 1 report materials"""

    print("="*60)
    print("Task 1 Report Generation")
    print("="*60)

    output_dir = "assets"

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Check if results exist
    if not os.path.exists("results/task1/results_summary.json"):
        print("Error: Task 1 results not found!")
        print("Please run task1_train.py first.")
        return

    # Load results
    results_summary, label_encoder, y_val = load_results()

    # Generate confusion matrix
    best_model = results_summary['best_model']
    generate_confusion_matrix(best_model, y_val, label_encoder, output_dir)

    # Generate model comparison
    generate_model_comparison(results_summary, output_dir)

    # Final summary
    print("\n" + "="*60)
    print("Task 1 Report Generation Complete!")
    print("="*60)
    print(f"\nGenerated files in {output_dir}/:")
    print(f"  - task1_confusion_matrix_{best_model.lower()}.png")
    print("  - task1_model_comparison.png")
    print(f"\nPerformance Summary:")
    for model_name, perf in results_summary['model_performance'].items():
        print(f"  {model_name}: Top-1={perf['top1_accuracy']:.4f}, Top-3={perf['top3_accuracy']:.4f}")
    print(f"\nBest Model: {best_model}")
    print("="*60)


if __name__ == "__main__":
    main()