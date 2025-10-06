import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import librosa
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import warnings
import pickle

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Artist20Dataset(Dataset):
    def __init__(
        self,
        json_file_path,
        base_path="data/artist20",
        sr=16000,
        duration=30,
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

        self.label_encoder = LabelEncoder()
        self.encoded_labels = self.label_encoder.fit_transform(self.labels)

        print(
            f"Dataset: {len(self.file_paths)} files, {len(self.label_encoder.classes_)} artists"
        )

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
                if self.augment:
                    start = np.random.randint(0, len(audio) - self.samples)
                    audio = audio[start : start + self.samples]
                else:
                    audio = audio[: self.samples]
            else:
                audio = np.pad(audio, (0, self.samples - len(audio)), mode="constant")

            # Simple augmentation
            if self.augment and np.random.random() < 0.3:
                audio = audio + np.random.normal(0, 0.003, audio.shape)

            # Mel spectrogram with fewer bins for speed
            mel_spec = librosa.feature.melspectrogram(
                y=audio, sr=self.sr, n_mels=64, fmax=8000, hop_length=512, n_fft=1024
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

            # Normalize
            mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (
                mel_spec_db.std() + 1e-8
            )

            return torch.FloatTensor(mel_spec_db).unsqueeze(0), self.encoded_labels[idx]

        except Exception as e:
            print(f"Error loading {full_path}: {e}")
            empty_mel = np.zeros((64, int(self.samples / 512) + 1))
            return torch.FloatTensor(empty_mel).unsqueeze(0), self.encoded_labels[idx]


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


def mixup_data(x, y, alpha=0.2):
    """Apply mixup augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_epoch(
    model, train_loader, criterion, optimizer, device, use_mixup=True, scaler=None
):
    model.train()
    train_loss = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(train_loader, desc="Training", colour="blue", ncols=100)
    for audio, labels in pbar:
        audio, labels = (
            audio.to(device, non_blocking=True),
            labels.to(device, non_blocking=True),
        )

        optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()

        # Mixed precision training
        if scaler is not None:
            with torch.amp.autocast("cuda"):
                # Apply mixup with 50% probability
                if use_mixup and np.random.random() < 0.5:
                    audio, labels_a, labels_b, lam = mixup_data(
                        audio, labels, alpha=0.2
                    )
                    outputs = model(audio)
                    loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                else:
                    outputs = model(audio)
                    loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Apply mixup with 50% probability
            if use_mixup and np.random.random() < 0.5:
                audio, labels_a, labels_b, lam = mixup_data(audio, labels, alpha=0.2)
                outputs = model(audio)
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            else:
                outputs = model(audio)
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix({"loss": f"{loss.item():.3f}"})

    avg_loss = train_loss / len(train_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy


def validate_epoch(model, val_loader, criterion, device, scaler=None):
    model.eval()
    val_loss = 0.0
    all_preds_top1 = []
    all_preds_top3 = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation", colour="green", ncols=100)
        for audio, labels in pbar:
            audio, labels = (
                audio.to(device, non_blocking=True),
                labels.to(device, non_blocking=True),
            )

            # Use mixed precision for validation too
            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    outputs = model(audio)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(audio)
                loss = criterion(outputs, labels)

            val_loss += loss.item()

            _, predicted_top1 = torch.max(outputs.data, 1)
            all_preds_top1.extend(predicted_top1.cpu().numpy())

            _, predicted_top3 = torch.topk(outputs.data, 3, dim=1)
            all_preds_top3.extend(predicted_top3.cpu().numpy())

            all_labels.extend(labels.cpu().numpy())
            pbar.set_postfix({"loss": f"{loss.item():.3f}"})

    avg_loss = val_loss / len(val_loader)
    top1_acc = accuracy_score(all_labels, all_preds_top1)
    top3_acc = sum(
        1 for i, label in enumerate(all_labels) if label in all_preds_top3[i]
    ) / len(all_labels)

    return avg_loss, top1_acc, top3_acc


def main():
    os.makedirs("results/task2", exist_ok=True)

    print("=" * 70)
    print("Task 2: ResNet-based CNN Artist Classifier")
    print("=" * 70)

    # Optimized parameters
    DURATION = 150  # Extended duration for more audio information
    EPOCHS = 100
    BATCH_SIZE = 32  # Increased to maximize GPU utilization
    LEARNING_RATE = 0.01  # Much higher LR for faster convergence
    PATIENCE = 15
    RESUME_TRAINING = True  # Set to True to resume from checkpoint
    CHECKPOINT_PATH = "results/task2/best_resnet_model_150s.pth"
    LABEL_ENCODER_PATH = "results/task2/label_encoder_resnet_150s.pkl"

    print("\nConfiguration:")
    print(
        f"  Duration: {DURATION}s | Batch: {BATCH_SIZE} | LR: {LEARNING_RATE} | Patience: {PATIENCE}"
    )
    print(f"  Resume Training: {RESUME_TRAINING}")

    # Load datasets
    print("\nLoading datasets...")

    # Check if resuming and label encoder exists
    if RESUME_TRAINING and os.path.exists(LABEL_ENCODER_PATH):
        print(f"  Loading existing label encoder from {LABEL_ENCODER_PATH}")
        with open(LABEL_ENCODER_PATH, "rb") as f:
            label_encoder = pickle.load(f)

        train_dataset = Artist20Dataset(
            "data/artist20/train.json", duration=DURATION, augment=True
        )
        train_dataset.label_encoder = label_encoder
        train_dataset.encoded_labels = label_encoder.transform(train_dataset.labels)

        val_dataset = Artist20Dataset(
            "data/artist20/val.json", duration=DURATION, augment=False
        )
        val_dataset.label_encoder = label_encoder
        val_dataset.encoded_labels = label_encoder.transform(val_dataset.labels)
    else:
        train_dataset = Artist20Dataset(
            "data/artist20/train.json", duration=DURATION, augment=True
        )
        val_dataset = Artist20Dataset(
            "data/artist20/val.json", duration=DURATION, augment=False
        )

        val_dataset.label_encoder = train_dataset.label_encoder
        val_dataset.encoded_labels = train_dataset.label_encoder.transform(
            val_dataset.labels
        )

        with open(LABEL_ENCODER_PATH, "wb") as f:
            pickle.dump(train_dataset.label_encoder, f)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8,  # Increased workers for faster data loading
        pin_memory=True,
        prefetch_factor=4,  # Prefetch more batches
        persistent_workers=True,  # Keep workers alive between epochs
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=8,  # Increased workers for faster data loading
        pin_memory=True,
        prefetch_factor=4,  # Prefetch more batches
        persistent_workers=True,  # Keep workers alive between epochs
    )

    print(
        f"  Train: {len(train_dataset)} | Val: {len(val_dataset)} | Classes: {len(train_dataset.label_encoder.classes_)}"
    )

    # Create model
    num_classes = len(train_dataset.label_encoder.classes_)
    model = ImprovedCNN(num_classes).to(device)

    # Enable mixed precision training for better GPU utilization
    scaler = torch.amp.GradScaler("cuda") if torch.cuda.is_available() else None

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {total_params:,} parameters | Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Mixed Precision: {'Enabled' if scaler else 'Disabled'}")

    # Training setup with label smoothing for better generalization
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)

    # Cosine annealing with warmup for better convergence
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    best_val_score = 0.0
    epochs_no_improve = 0
    start_epoch = 0

    # Load checkpoint if resuming (BEFORE compiling)
    if RESUME_TRAINING and os.path.exists(CHECKPOINT_PATH):
        print(f"\n{'=' * 70}")
        print(f"Resuming training from checkpoint: {CHECKPOINT_PATH}")
        print(f"{'=' * 70}")

        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        # Load optimizer and scheduler if available
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print("  ✓ Loaded optimizer state")

        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            print("  ✓ Loaded scheduler state")

        # Load training state
        best_val_score = checkpoint.get("val_score", 0.0)
        start_epoch = checkpoint.get("epoch", 0)

        print(f"  ✓ Loaded model from epoch {start_epoch}")
        print(f"  ✓ Previous best score: {best_val_score:.4f}")
        print(f"  ✓ Previous top-1: {checkpoint.get('val_top1', 0.0):.4f}")
        print(f"  ✓ Previous top-3: {checkpoint.get('val_top3', 0.0):.4f}")
        print(f"\n  Continuing from epoch {start_epoch + 1}...")
    else:
        print(f"\n{'=' * 70}")
        print("Starting training from scratch...")
        print(f"{'=' * 70}")

    # Compile model AFTER loading checkpoint (PyTorch 2.0+)
    try:
        model = torch.compile(model)
        print("  ✓ Model compiled with torch.compile()")
    except Exception as e:
        print(f"  ⚠ torch.compile() not available, using standard model: {e}")

    print(f"\n{'=' * 70}")
    print("Starting training...")
    print(f"{'=' * 70}\n")

    for epoch in range(start_epoch, EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler=scaler
        )

        # Validate
        val_loss, val_top1, val_top3 = validate_epoch(
            model, val_loader, criterion, device, scaler=scaler
        )

        val_score = val_top1 + val_top3 * 0.5

        # Print summary
        print(f"  Train: loss={train_loss:.4f} acc={train_acc:.4f}")
        print(
            f"  Val:   loss={val_loss:.4f} top1={val_top1:.4f} top3={val_top3:.4f} score={val_score:.4f}"
        )

        # Save best
        if val_score > best_val_score:
            best_val_score = val_score
            epochs_no_improve = 0
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_score": val_score,
                    "val_top1": val_top1,
                    "val_top3": val_top3,
                },
                CHECKPOINT_PATH,
            )
            print(f"  🏆 New best! score={best_val_score:.4f}")
        else:
            epochs_no_improve += 1
            print(
                f"  No improve: {epochs_no_improve}/{PATIENCE} (best={best_val_score:.4f})"
            )

        # LR scheduling (cosine annealing - step every epoch)
        scheduler.step()

        # Early stopping
        if epochs_no_improve >= PATIENCE:
            print(f"\n⏹️  Early stopping at epoch {epoch + 1}")
            break

        print()

    print(f"\n{'=' * 70}")
    print(f"Training complete! Best score: {best_val_score:.4f}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
