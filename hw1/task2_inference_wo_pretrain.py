import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import librosa
import os
from tqdm import tqdm
import warnings
import pickle

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestDataset(Dataset):
    def __init__(self, test_dir="data/artist20/test", sr=16000, duration=150):
        self.test_dir = test_dir
        self.sr = sr
        self.duration = duration
        self.samples = int(sr * duration)

        # Get all test files
        self.test_files = sorted(
            [f for f in os.listdir(test_dir) if f.endswith(".mp3")]
        )
        print(f"Found {len(self.test_files)} test files")

    def __len__(self):
        return len(self.test_files)

    def __getitem__(self, idx):
        file_name = self.test_files[idx]
        file_path = os.path.join(self.test_dir, file_name)

        try:
            # Load audio
            audio, _ = librosa.load(file_path, sr=self.sr)

            # Crop or pad
            if len(audio) > self.samples:
                audio = audio[: self.samples]
            else:
                audio = np.pad(audio, (0, self.samples - len(audio)), mode="constant")

            # Mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio, sr=self.sr, n_mels=64, fmax=8000, hop_length=512, n_fft=1024
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

            # Normalize
            mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (
                mel_spec_db.std() + 1e-8
            )

            return torch.FloatTensor(mel_spec_db).unsqueeze(0), file_name

        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            empty_mel = np.zeros((64, int(self.samples / 512) + 1))
            return torch.FloatTensor(empty_mel).unsqueeze(0), file_name


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


def main():
    print("=" * 70)
    print("Task 2: ResNet-based CNN Artist Classifier - Inference")
    print("=" * 70)

    # Configuration - must match training
    DURATION = 150
    BATCH_SIZE = 32
    CHECKPOINT_PATH = "results/task2/best_resnet_model_150s.pth"
    LABEL_ENCODER_PATH = "results/task2/label_encoder_resnet_150s.pkl"
    OUTPUT_JSON = "r14921046.json"

    print("\nConfiguration:")
    print(f"  Duration: {DURATION}s | Batch: {BATCH_SIZE}")
    print(f"  Checkpoint: {CHECKPOINT_PATH}")
    print(f"  Label Encoder: {LABEL_ENCODER_PATH}")
    print(f"  Output: {OUTPUT_JSON}")

    # Load label encoder
    print("\nLoading label encoder...")
    with open(LABEL_ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)
    print(f"  Classes: {len(label_encoder.classes_)}")
    print(f"  Artists: {', '.join(label_encoder.classes_[:5])}...")

    # Load test dataset
    print("\nLoading test dataset...")
    test_dataset = TestDataset(duration=DURATION)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
    )
    print(f"  Test files: {len(test_dataset)}")

    # Load model
    print("\nLoading model...")
    num_classes = len(label_encoder.classes_)
    model = ImprovedCNN(num_classes).to(device)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    # Handle torch.compile() prefix
    state_dict = checkpoint["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")
    print(f"  Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    print(f"\n  ✓ Loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"  ✓ Val top-1: {checkpoint.get('val_top1', 0.0):.4f}")
    print(f"  ✓ Val top-3: {checkpoint.get('val_top3', 0.0):.4f}")

    # Run inference
    print("\n" + "=" * 70)
    print("Running inference...")
    print("=" * 70 + "\n")

    predictions = {}

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Inference", colour="cyan", ncols=100)
        for audio, file_names in pbar:
            audio = audio.to(device, non_blocking=True)

            # Use mixed precision for inference
            if torch.cuda.is_available():
                with torch.amp.autocast("cuda"):
                    outputs = model(audio)
            else:
                outputs = model(audio)

            # Get top-3 predictions
            _, top3_indices = torch.topk(outputs, 3, dim=1)
            top3_indices = top3_indices.cpu().numpy()

            # Convert to artist names
            for i, file_name in enumerate(file_names):
                file_id = file_name.replace(".mp3", "")
                top3_artists = label_encoder.inverse_transform(top3_indices[i])
                predictions[file_id] = top3_artists.tolist()

    # Save predictions
    print(f"\nSaving predictions to {OUTPUT_JSON}...")
    with open(OUTPUT_JSON, "w") as f:
        json.dump(predictions, f, indent=4)

    print(f"\n{'=' * 70}")
    print("Inference complete!")
    print(f"  Total predictions: {len(predictions)}")
    print(f"  Output saved to: {OUTPUT_JSON}")
    print(f"{'=' * 70}")

    # Show sample predictions
    print("\nSample predictions:")
    for i, (file_id, artists) in enumerate(list(predictions.items())[:5]):
        print(f"  {file_id}.mp3 → {', '.join(artists)}")


if __name__ == "__main__":
    main()
