"""
Generate t-SNE visualizations for ResNet models
Compares epoch 1 vs final trained model
"""

import torch
import torch.nn as nn
import numpy as np
import json
import librosa
import os
import pickle
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


# ============================================================================
# Model Definition (same as training script)
# ============================================================================


class ResidualBlock(nn.Module):
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
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
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

        x_avg = self.avgpool(x)
        x_max = self.maxpool(x)
        x = torch.cat([x_avg, x_max], dim=1)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def extract_features(self, x):
        """Extract features before the final classifier"""
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x_avg = self.avgpool(x)
        x_max = self.maxpool(x)
        x = torch.cat([x_avg, x_max], dim=1)

        features = x.view(x.size(0), -1)
        return features


# ============================================================================
# Feature Extraction
# ============================================================================


def load_audio_and_extract_mel(file_path, sr=16000, duration=150):
    """Load audio and extract mel spectrogram"""
    try:
        audio, _ = librosa.load(file_path, sr=sr)

        samples = int(sr * duration)
        if len(audio) > samples:
            audio = audio[:samples]
        else:
            audio = np.pad(audio, (0, samples - len(audio)), mode="constant")

        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_mels=64, fmax=8000, hop_length=512, n_fft=1024
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)

        return torch.FloatTensor(mel_spec_db).unsqueeze(0)

    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def extract_features_from_model(
    model, val_json_path, label_encoder, device, base_path="data/artist20", max_samples=None
):
    """Extract features from validation set using the model"""

    model.eval()

    with open(val_json_path, "r") as f:
        val_file_paths = json.load(f)

    if max_samples:
        val_file_paths = val_file_paths[:max_samples]

    features_list = []
    labels_list = []

    print(f"Extracting features from {len(val_file_paths)} files...")

    with torch.no_grad():
        for file_path in tqdm(val_file_paths):
            full_path = os.path.join(base_path, file_path)

            # Get label
            artist_name = file_path.split("/")[2]

            # Load and process audio
            mel_spec = load_audio_and_extract_mel(full_path)
            if mel_spec is None:
                continue

            mel_spec = mel_spec.unsqueeze(0).to(device)

            # Extract features
            features = model.extract_features(mel_spec)
            features_list.append(features.cpu().numpy()[0])
            labels_list.append(artist_name)

    features_array = np.array(features_list)
    labels_encoded = label_encoder.transform(labels_list)

    return features_array, labels_encoded, labels_list


# ============================================================================
# t-SNE Visualization
# ============================================================================


def plot_tsne(
    features_epoch1,
    features_final,
    labels,
    label_encoder,
    output_path="assets/task2_tsne_comparison.png",
):
    """Create side-by-side t-SNE plots"""

    print("Computing t-SNE for epoch 1 model...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    tsne_epoch1 = tsne.fit_transform(features_epoch1)

    print("Computing t-SNE for final model...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    tsne_final = tsne.fit_transform(features_final)

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Color palette
    n_classes = len(label_encoder.classes_)
    colors = sns.color_palette("husl", n_classes)

    # Plot epoch 1
    for i, artist in enumerate(label_encoder.classes_):
        mask = labels == i
        axes[0].scatter(
            tsne_epoch1[mask, 0],
            tsne_epoch1[mask, 1],
            c=[colors[i]],
            label=artist,
            alpha=0.6,
            s=50,
        )

    axes[0].set_title("t-SNE: ResNet Epoch 1 (Random Init)", fontsize=16, fontweight="bold")
    axes[0].set_xlabel("t-SNE Component 1", fontsize=12)
    axes[0].set_ylabel("t-SNE Component 2", fontsize=12)
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Plot final model
    for i, artist in enumerate(label_encoder.classes_):
        mask = labels == i
        axes[1].scatter(
            tsne_final[mask, 0],
            tsne_final[mask, 1],
            c=[colors[i]],
            label=artist,
            alpha=0.6,
            s=50,
        )

    axes[1].set_title("t-SNE: ResNet Final Model (Trained)", fontsize=16, fontweight="bold")
    axes[1].set_xlabel("t-SNE Component 1", fontsize=12)
    axes[1].set_ylabel("t-SNE Component 2", fontsize=12)
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"t-SNE comparison saved to {output_path}")


# ============================================================================
# Main Function
# ============================================================================


def main():
    print("=" * 60)
    print("t-SNE Visualization for ResNet Models")
    print("=" * 60)

    # Configuration
    model_epoch1_path = "results/task2/best_resnet_model_150s_1.pth"
    label_encoder_epoch1_path = "results/task2/label_encoder_resnet_150s_1.pkl"

    model_final_path = "results/task2/best_resnet_model_150s.pth"
    label_encoder_final_path = "results/task2/label_encoder_resnet_150s.pkl"

    val_json_path = "data/artist20/val.json"
    output_dir = "assets"

    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load label encoders
    print("\n" + "=" * 60)
    print("Loading label encoders...")
    print("=" * 60)

    with open(label_encoder_final_path, "rb") as f:
        label_encoder = pickle.load(f)

    print(f"Label encoder loaded with {len(label_encoder.classes_)} classes")

    # Load models
    print("\n" + "=" * 60)
    print("Loading models...")
    print("=" * 60)

    num_classes = len(label_encoder.classes_)

    # Helper function to remove _orig_mod prefix from compiled models
    def remove_orig_mod_prefix(state_dict):
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("_orig_mod."):
                new_key = key.replace("_orig_mod.", "")
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        return new_state_dict

    # Epoch 1 model
    model_epoch1 = ImprovedCNN(num_classes).to(device)
    checkpoint_epoch1 = torch.load(model_epoch1_path, map_location=device)
    if "model_state_dict" in checkpoint_epoch1:
        state_dict = remove_orig_mod_prefix(checkpoint_epoch1["model_state_dict"])
        model_epoch1.load_state_dict(state_dict)
    else:
        state_dict = remove_orig_mod_prefix(checkpoint_epoch1)
        model_epoch1.load_state_dict(state_dict)
    model_epoch1.eval()
    print(f"Epoch 1 model loaded from {model_epoch1_path}")

    # Final model
    model_final = ImprovedCNN(num_classes).to(device)
    checkpoint_final = torch.load(model_final_path, map_location=device)
    if "model_state_dict" in checkpoint_final:
        state_dict = remove_orig_mod_prefix(checkpoint_final["model_state_dict"])
        model_final.load_state_dict(state_dict)
    else:
        state_dict = remove_orig_mod_prefix(checkpoint_final)
        model_final.load_state_dict(state_dict)
    model_final.eval()
    print(f"Final model loaded from {model_final_path}")

    # Extract features
    print("\n" + "=" * 60)
    print("Extracting features from epoch 1 model...")
    print("=" * 60)
    features_epoch1, labels, _ = extract_features_from_model(
        model_epoch1, val_json_path, label_encoder, device
    )

    print("\n" + "=" * 60)
    print("Extracting features from final model...")
    print("=" * 60)
    features_final, _, _ = extract_features_from_model(
        model_final, val_json_path, label_encoder, device
    )

    # Generate t-SNE visualization
    print("\n" + "=" * 60)
    print("Generating t-SNE visualization...")
    print("=" * 60)
    plot_tsne(
        features_epoch1,
        features_final,
        labels,
        label_encoder,
        output_path=os.path.join(output_dir, "task2_tsne_comparison.png"),
    )

    print("\n" + "=" * 60)
    print("t-SNE visualization complete!")
    print("=" * 60)
    print(f"Output saved to {output_dir}/task2_tsne_comparison.png")


if __name__ == "__main__":
    main()
