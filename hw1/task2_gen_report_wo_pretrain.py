"""
Task 2 Report Generation Script (Without Pretrain)
Generates visualizations for ResNet-based CNN without pretrained features
"""

import torch
import torch.nn as nn
import numpy as np
import json
import librosa
import os
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import pickle

warnings.filterwarnings("ignore")

# Set style
plt.style.use("default")
sns.set_palette("husl")


# ============================================================================
# Model Definition (must match training script)
# ============================================================================


class ResidualBlock(nn.Module):
    """Residual block for better gradient flow"""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class ImprovedCNN(nn.Module):
    """
    Improved CNN with residual connections and attention
    Better learning dynamics for artist classification
    """

    def __init__(self, num_classes):
        super().__init__()

        # Initial conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)

        # Global pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))

        # Classifier with stronger regularization
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),  # 256 from avg + 256 from max
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # Dual pooling
        x_avg = self.avgpool(x)
        x_max = self.maxpool(x)
        x = torch.cat([x_avg, x_max], dim=1)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class Artist20Dataset(Dataset):
    def __init__(
        self,
        json_file_path,
        base_path="data/artist20",
        sr=16000,
        duration=150,
        augment=False,
    ):
        self.sr = sr
        self.duration = duration
        self.samples = int(sr * duration)
        self.base_path = base_path
        self.augment = augment

        with open(json_file_path, "r") as f:
            self.file_paths = json.load(f)

        # Extract labels
        self.labels = []
        for file_path in self.file_paths:
            artist_name = file_path.split("/")[2]
            self.labels.append(artist_name)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        full_path = os.path.join(self.base_path, file_path)

        try:
            # Load audio
            audio, _ = librosa.load(full_path, sr=self.sr)

            # Crop or pad
            if len(audio) > self.samples:
                audio = audio[: self.samples]
            else:
                audio = np.pad(audio, (0, self.samples - len(audio)), mode="constant")

            # Mel spectrogram with fewer bins for speed
            mel_spec = librosa.feature.melspectrogram(
                y=audio, sr=self.sr, n_mels=64, fmax=8000, hop_length=512, n_fft=1024
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

            # Normalize
            mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (
                mel_spec_db.std() + 1e-8
            )

            return torch.FloatTensor(mel_spec_db).unsqueeze(0), self.labels[idx]

        except Exception as e:
            print(f"Error loading {full_path}: {e}")
            empty_mel = np.zeros((64, int(self.samples / 512) + 1))
            return torch.FloatTensor(empty_mel).unsqueeze(0), self.labels[idx]


# ============================================================================
# Evaluation Functions
# ============================================================================


def load_model_and_encoder(model_path, label_encoder_path):
    """Load the trained model and label encoder"""

    print(f"\nLoading label encoder from {label_encoder_path}...")
    with open(label_encoder_path, "rb") as f:
        label_encoder = pickle.load(f)

    print(f"  Classes: {len(label_encoder.classes_)}")

    num_classes = len(label_encoder.classes_)
    model = ImprovedCNN(num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nLoading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)

    # Handle torch.compile() prefix
    state_dict = checkpoint["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    print(f"  Model loaded on {device}")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  Val Top-1: {checkpoint.get('val_top1', 0.0):.4f}")
    print(f"  Val Top-3: {checkpoint.get('val_top3', 0.0):.4f}")

    return model, label_encoder, device


def evaluate_validation_set(model, label_encoder, device, val_json_path):
    """Evaluate model on validation set and compute top-1 and top-3 accuracy"""

    print("\nEvaluating on validation set...")

    val_dataset = Artist20Dataset(val_json_path, duration=150, augment=False)
    val_dataset.label_encoder = label_encoder
    val_dataset.encoded_labels = label_encoder.transform(val_dataset.labels)

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    all_preds_top1 = []
    all_preds_top3 = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation", colour="green")
        for audio, labels in pbar:
            audio = audio.to(device, non_blocking=True)

            # Forward pass
            if torch.cuda.is_available():
                with torch.amp.autocast("cuda"):
                    outputs = model(audio)
            else:
                outputs = model(audio)

            # Top-1 predictions
            _, predicted_top1 = torch.max(outputs.data, 1)
            all_preds_top1.extend(predicted_top1.cpu().numpy())

            # Top-3 predictions
            _, predicted_top3 = torch.topk(outputs.data, 3, dim=1)
            all_preds_top3.extend(predicted_top3.cpu().numpy())

            # True labels
            encoded_labels = label_encoder.transform(labels)
            all_labels.extend(encoded_labels)

    # Calculate accuracies
    top1_accuracy = accuracy_score(all_labels, all_preds_top1)
    top3_accuracy = sum(
        1 for i, label in enumerate(all_labels) if label in all_preds_top3[i]
    ) / len(all_labels)

    print(f"\n{'=' * 70}")
    print("Validation Set Results:")
    print(f"{'=' * 70}")
    print(f"Val Top-1 Accuracy: {top1_accuracy:.4f} ({top1_accuracy * 100:.2f}%)")
    print(f"Val Top-3 Accuracy: {top3_accuracy:.4f} ({top3_accuracy * 100:.2f}%)")
    print(f"{'=' * 70}")

    return top1_accuracy, top3_accuracy, all_preds_top1, all_labels


def generate_confusion_matrix(model, label_encoder, device, val_json_path, output_dir):
    """Generate confusion matrix for validation set"""

    print("\nGenerating confusion matrix...")

    val_dataset = Artist20Dataset(val_json_path, duration=150, augment=False)
    val_dataset.label_encoder = label_encoder
    val_dataset.encoded_labels = label_encoder.transform(val_dataset.labels)

    val_loader = DataLoader(
        val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
    )

    all_predictions = []
    all_true_labels = []

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Making predictions")
        for audio, labels in pbar:
            audio = audio.to(device, non_blocking=True)

            if torch.cuda.is_available():
                with torch.amp.autocast("cuda"):
                    outputs = model(audio)
            else:
                outputs = model(audio)

            _, predicted = torch.max(outputs.data, 1)

            all_predictions.extend(predicted.cpu().numpy())
            encoded_labels = label_encoder.transform(labels)
            all_true_labels.extend(encoded_labels)

    cm = confusion_matrix(all_true_labels, all_predictions)

    plt.figure(figsize=(16, 12))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_,
    )
    plt.title(
        "Confusion Matrix - ResNet CNN (No Pretrain)",
        fontsize=16,
        fontweight="bold",
    )
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("Actual", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "task2_confusion_matrix_resnet.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(f"Confusion matrix saved to {output_dir}/task2_confusion_matrix_resnet.png")

    # Print classification report
    print("\nClassification Report:")
    print(
        classification_report(
            all_true_labels,
            all_predictions,
            target_names=label_encoder.classes_,
            zero_division=0,
        )
    )


# ============================================================================
# Chart Generation Functions
# ============================================================================


def create_architecture_diagram(output_dir):
    """Create architecture diagram for ResNet-based CNN"""

    print("Generating architecture diagram...")

    fig, ax = plt.subplots(figsize=(14, 10))

    # Define boxes for architecture
    boxes = [
        {"name": "Audio Input\n(150s, 16kHz)", "pos": (1, 7), "color": "#E8F4FD"},
        {"name": "Mel Spectrogram\n(64 mels)", "pos": (3, 7), "color": "#D1ECF1"},
        {
            "name": "Conv2d (7×7)\n64 channels\nstride=2",
            "pos": (5, 7),
            "color": "#BEE5EB",
        },
        {"name": "MaxPool2d\n(3×3)", "pos": (5, 6), "color": "#BEE5EB"},
        {
            "name": "ResBlock×2\n64 channels",
            "pos": (7, 7),
            "color": "#85C1E9",
        },
        {
            "name": "ResBlock×2\n128 channels\nstride=2",
            "pos": (7, 5.5),
            "color": "#5DADE2",
        },
        {
            "name": "ResBlock×2\n256 channels\nstride=2",
            "pos": (7, 4),
            "color": "#3498DB",
        },
        {"name": "AdaptiveAvgPool", "pos": (9, 6.5), "color": "#2E86AB"},
        {"name": "AdaptiveMaxPool", "pos": (9, 4.5), "color": "#2E86AB"},
        {"name": "Concatenate\n512-dim", "pos": (11, 5.5), "color": "#1B4F72"},
        {"name": "Dropout (0.5)", "pos": (11, 4.5), "color": "#85C1E9"},
        {"name": "Linear (512→256)", "pos": (13, 5.5), "color": "#5DADE2"},
        {"name": "BatchNorm + ReLU", "pos": (13, 4.5), "color": "#3498DB"},
        {"name": "Dropout (0.3)", "pos": (13, 3.5), "color": "#85C1E9"},
        {"name": "Linear (256→20)", "pos": (15, 4.5), "color": "#2E86AB"},
        {"name": "Artist\nPrediction", "pos": (17, 4.5), "color": "#1B4F72"},
    ]

    for box in boxes:
        rect = plt.Rectangle(
            (box["pos"][0] - 0.5, box["pos"][1] - 0.3),
            1.0,
            0.6,
            facecolor=box["color"],
            edgecolor="black",
            linewidth=1.5,
        )
        ax.add_patch(rect)
        ax.text(
            box["pos"][0],
            box["pos"][1],
            box["name"],
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
        )

    # Arrows
    arrow_props = dict(arrowstyle="->", lw=2, color="black")
    arrows = [
        ((1.5, 7), (2.5, 7)),
        ((3.5, 7), (4.5, 7)),
        ((5, 6.7), (5, 6.3)),
        ((5.5, 6), (6.5, 7)),
        ((7.5, 6.75), (6.5, 5.75)),
        ((7.5, 5.25), (6.5, 4.25)),
        ((7.5, 5.5), (8.5, 6.25)),
        ((7.5, 4), (8.5, 4.75)),
        ((9.5, 6.5), (10.5, 5.75)),
        ((9.5, 4.5), (10.5, 5.25)),
        ((11, 5.2), (11, 4.8)),
        ((11.5, 5.5), (12.5, 5.5)),
        ((13, 5.2), (13, 4.8)),
        ((13, 4.2), (13, 3.8)),
        ((13.5, 4.5), (14.5, 4.5)),
        ((15.5, 4.5), (16.5, 4.5)),
    ]

    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops=arrow_props)

    ax.set_xlim(0, 18)
    ax.set_ylim(2.5, 8)
    ax.set_title(
        "ResNet-based CNN Architecture (No Pretrain)",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "task2_architecture_diagram_resnet.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(
        f"Architecture diagram saved to {output_dir}/task2_architecture_diagram_resnet.png"
    )


def create_model_comparison_chart(task1_results, task2_panns, task2_resnet, output_dir):
    """Create comparison between all models"""

    print("Generating comprehensive model comparison chart...")

    models = [
        "Traditional ML\n(SVM)",
        "Deep Learning\n(PANNs)",
        "Deep Learning\n(ResNet CNN)",
    ]
    top1_acc = [
        task1_results["top1"] * 100,
        task2_panns["top1"] * 100,
        task2_resnet["top1"] * 100,
    ]
    top3_acc = [
        task1_results["top3"] * 100,
        task2_panns["top3"] * 100,
        task2_resnet["top3"] * 100,
    ]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 7))

    bars1 = ax.bar(
        x - width / 2,
        top1_acc,
        width,
        label="Top-1 Accuracy",
        color="#2E86AB",
        alpha=0.8,
    )
    bars2 = ax.bar(
        x + width / 2,
        top3_acc,
        width,
        label="Top-3 Accuracy",
        color="#A23B72",
        alpha=0.8,
    )

    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(
        "Model Performance Comparison: Traditional ML vs Deep Learning",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 90)

    for bar in bars1:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.5,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    for bar in bars2:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.5,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "task2_all_models_comparison.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(
        f"Model comparison chart saved to {output_dir}/task2_all_models_comparison.png"
    )


# ============================================================================
# Main Function
# ============================================================================


def main():
    """Main function to generate all Task 2 (without pretrain) report materials"""

    print("=" * 70)
    print("Task 2 Report Generation - ResNet CNN (No Pretrain)")
    print("=" * 70)

    # Configuration
    model_path = "results/task2/best_resnet_model_150s.pth"
    label_encoder_path = "results/task2/label_encoder_resnet_150s.pkl"
    val_json_path = "data/artist20/val.json"
    output_dir = "assets"

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("results/task2", exist_ok=True)

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please train the model first using task2_train_wo_pretrain.py")
        return

    # Step 1: Load model
    print("\n" + "=" * 70)
    print("Step 1: Loading Model")
    print("=" * 70)
    model, label_encoder, device = load_model_and_encoder(
        model_path, label_encoder_path
    )

    # Step 2: Evaluate on validation set
    print("\n" + "=" * 70)
    print("Step 2: Evaluating on Validation Set")
    print("=" * 70)
    top1_acc, top3_acc, predictions, true_labels = evaluate_validation_set(
        model, label_encoder, device, val_json_path
    )

    # Save evaluation results
    results = {
        "val_top1_accuracy": float(top1_acc),
        "val_top3_accuracy": float(top3_acc),
        "model_path": model_path,
    }

    with open("results/task2/val_set_evaluation_resnet.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to results/task2/val_set_evaluation_resnet.json")

    # Step 3: Generate confusion matrix
    print("\n" + "=" * 70)
    print("Step 3: Generating Confusion Matrix")
    print("=" * 70)
    generate_confusion_matrix(model, label_encoder, device, val_json_path, output_dir)

    # Step 4: Load Task 1 and Task 2 (PANNs) results for comparison
    print("\n" + "=" * 70)
    print("Step 4: Loading Previous Results for Comparison")
    print("=" * 70)

    # Load Task 1 results
    if os.path.exists("results/task1/results_summary.json"):
        with open("results/task1/results_summary.json", "r") as f:
            task1_data = json.load(f)

        task1_results = {
            "top1": task1_data["model_performance"]["SVM"]["top1_accuracy"],
            "top3": task1_data["model_performance"]["SVM"]["top3_accuracy"],
        }
        print(
            f"Task 1 (SVM): Top-1={task1_results['top1']:.4f}, Top-3={task1_results['top3']:.4f}"
        )
    else:
        task1_results = {"top1": 0.5714, "top3": 0.7879}
        print("Task 1 results not found, using default values")

    # Load Task 2 (PANNs) results
    if os.path.exists("results/task2/val_set_evaluation.json"):
        with open("results/task2/val_set_evaluation.json", "r") as f:
            task2_panns_data = json.load(f)

        task2_panns = {
            "top1": task2_panns_data["val_top1_accuracy"],
            "top3": task2_panns_data["val_top3_accuracy"],
        }
        print(
            f"Task 2 (PANNs): Top-1={task2_panns['top1']:.4f}, Top-3={task2_panns['top3']:.4f}"
        )
    else:
        task2_panns = {"top1": 0.6017, "top3": 0.8312}
        print("Task 2 (PANNs) results not found, using default values")

    task2_resnet = {"top1": top1_acc, "top3": top3_acc}
    print(
        f"Task 2 (ResNet): Top-1={task2_resnet['top1']:.4f}, Top-3={task2_resnet['top3']:.4f}"
    )

    # Step 5: Generate all charts
    print("\n" + "=" * 70)
    print("Step 5: Generating Charts")
    print("=" * 70)

    # Check if architecture diagram already exists
    arch_diagram_path = os.path.join(
        output_dir, "task2_architecture_diagram_resnet.png"
    )
    if not os.path.exists(arch_diagram_path):
        print(
            "Note: Architecture diagram not found. Run gen_resnet_architecture.py to create it."
        )
        print("Skipping architecture diagram generation...")
    else:
        print(f"✓ Architecture diagram already exists: {arch_diagram_path}")

    create_model_comparison_chart(task1_results, task2_panns, task2_resnet, output_dir)

    # Final summary
    print("\n" + "=" * 70)
    print("Task 2 (No Pretrain) Report Generation Complete!")
    print("=" * 70)
    print(f"\nGenerated files in {output_dir}/:")
    print("  - task2_confusion_matrix_resnet.png")
    print("  - task2_architecture_diagram_resnet.png")
    print("  - task2_all_models_comparison.png")
    print("\nValidation Results:")
    print(f"  - Top-1 Accuracy: {top1_acc * 100:.2f}%")
    print(f"  - Top-3 Accuracy: {top3_acc * 100:.2f}%")
    print("\nComparison:")
    print(
        f"  - vs SVM: Top-1 {(top1_acc - task1_results['top1']) * 100:+.2f}%, Top-3 {(top3_acc - task1_results['top3']) * 100:+.2f}%"
    )
    print(
        f"  - vs PANNs: Top-1 {(top1_acc - task2_panns['top1']) * 100:+.2f}%, Top-3 {(top3_acc - task2_panns['top3']) * 100:+.2f}%"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
