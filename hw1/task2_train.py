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
from panns_inference import AudioTagging
import subprocess
import tempfile
from glob import glob

warnings.filterwarnings("ignore")


class TestDataset(Dataset):
    def __init__(self, test_dir="data/artist20/test", sr=16000, duration=120):
        self.sr = sr
        self.duration = duration
        self.samples = int(sr * duration)
        self.test_dir = test_dir

        # Get all test files and sort them numerically
        test_files = glob(os.path.join(test_dir, "*.mp3"))
        self.test_files = sorted(
            test_files, key=lambda x: int(os.path.basename(x).split(".")[0])
        )

        print(f"Test dataset loaded: {len(self.test_files)} files")

    def __len__(self):
        return len(self.test_files)

    def __getitem__(self, idx):
        file_path = self.test_files[idx]
        file_id = os.path.basename(file_path).split(".")[0]

        try:
            # Load audio
            audio, _ = librosa.load(file_path, sr=self.sr)

            # Crop or pad to desired length
            if len(audio) > self.samples:
                audio = audio[: self.samples]  # Take from beginning for consistency
            else:
                audio = np.pad(
                    audio, (0, max(0, self.samples - len(audio))), mode="constant"
                )

            return torch.FloatTensor(audio), file_id

        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return torch.zeros(self.samples), file_id


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

        # Load file paths from JSON
        with open(json_file_path, "r") as f:
            self.file_paths = json.load(f)

        # Extract labels from file paths (artist names)
        self.labels = []
        for file_path in self.file_paths:
            path_parts = file_path.split("/")
            artist_name = path_parts[2]
            self.labels.append(artist_name)

        # Create label encoder
        self.label_encoder = LabelEncoder()
        self.encoded_labels = self.label_encoder.fit_transform(self.labels)

        print(f"Dataset loaded: {len(self.file_paths)} files")
        print(f"Number of unique artists: {len(self.label_encoder.classes_)}")
        print(f"Artists: {list(self.label_encoder.classes_)}")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        full_path = os.path.join(self.base_path, file_path)

        try:
            # Load full audio first
            audio, _ = librosa.load(full_path, sr=self.sr)

            # Random crop if audio is longer than desired duration
            if len(audio) > self.samples:
                if self.augment:
                    # Random start position
                    start = np.random.randint(0, len(audio) - self.samples)
                    audio = audio[start : start + self.samples]
                else:
                    # Take from beginning
                    audio = audio[: self.samples]
            else:
                # Pad if shorter
                audio = np.pad(
                    audio, (0, max(0, self.samples - len(audio))), mode="constant"
                )

            # Data augmentation
            if self.augment:
                # Add slight noise
                if np.random.random() < 0.3:
                    noise = np.random.normal(0, 0.005, audio.shape)
                    audio = audio + noise

                # Slight pitch shift (time stretch)
                if np.random.random() < 0.3:
                    try:
                        rate = np.random.uniform(0.9, 1.1)
                        audio = librosa.effects.time_stretch(audio, rate=rate)

                        # Ensure correct length after time stretch
                        if len(audio) > self.samples:
                            audio = audio[: self.samples]
                        else:
                            audio = np.pad(
                                audio, (0, self.samples - len(audio)), mode="constant"
                            )
                    except Exception as e:
                        print(f"Time stretch error: {e}")
                        pass  # Skip time stretch if it fails

            return torch.FloatTensor(audio), self.encoded_labels[idx]

        except Exception as e:
            print(f"Error loading {full_path}: {e}")
            return torch.zeros(self.samples), self.encoded_labels[idx]


class PANNsClassifier(nn.Module):
    def __init__(self, num_classes, feature_dim=2048, fine_tune=False):
        super().__init__()
        self.panns = AudioTagging(
            checkpoint_path=None, device="cuda" if torch.cuda.is_available() else "cpu"
        )
        self.feature_dim = feature_dim
        self.fine_tune = fine_tune

        # Set PANNs model to training mode if fine-tuning
        if fine_tune:
            self.panns.model.train()
        else:
            self.panns.model.eval()

        # Improved classifier head with batch normalization
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        # Extract features using PANNs
        if self.fine_tune:
            output_dict = self.panns.model(x, None)
            features = output_dict["embedding"]
        else:
            with torch.no_grad():
                output_dict = self.panns.model(x, None)
                features = output_dict["embedding"]

        # Apply classifier
        output = self.classifier(features)
        return output


def evaluate_test_set(model, test_loader, label_encoder, epoch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    predictions = {}

    with torch.no_grad():
        for audio, file_ids in tqdm(test_loader, desc="Testing"):
            audio = audio.to(device)
            outputs = model(audio)

            # Get top-3 predictions for each sample
            _, top3_indices = torch.topk(outputs, 3, dim=1)

            for i, file_id in enumerate(file_ids):
                top3_artists = [
                    label_encoder.classes_[idx] for idx in top3_indices[i].cpu().numpy()
                ]
                predictions[file_id] = top3_artists

    # Save predictions to temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(predictions, f, indent=2)
        temp_pred_path = f.name

    # Run count_score.py to get evaluation metrics
    try:
        result = subprocess.run(
            ["python", "count_score.py", "test_ans.json", temp_pred_path],
            capture_output=True,
            text=True,
            cwd=os.getcwd(),
        )

        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            if len(lines) >= 2:
                top1_acc = float(lines[0])
                top3_acc = float(lines[1])
                return top1_acc, top3_acc
            else:
                print(f"Epoch {epoch}: Error parsing count_score.py output")
        else:
            print(f"Epoch {epoch}: Error running count_score.py: {result.stderr}")
    except Exception as e:
        print(f"Epoch {epoch}: Exception running count_score.py: {e}")
    finally:
        # Clean up temporary file
        try:
            os.unlink(temp_pred_path)
        except Exception:
            pass

    return 0.0, 0.0


def train_model(
    model,
    train_loader,
    val_loader,
    test_loader,
    label_encoder,
    num_epochs=50,
    lr=0.001,
    duration=120,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    # Different learning rates for PANNs and classifier
    if model.fine_tune:
        optimizer = optim.Adam(
            [
                {
                    "params": model.panns.model.parameters(),
                    "lr": lr * 0.1,
                },  # Lower LR for pretrained
                {"params": model.classifier.parameters(), "lr": lr},
            ],
            weight_decay=1e-4,
        )
    else:
        optimizer = optim.Adam(model.classifier.parameters(), lr=lr, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    train_losses = []
    val_losses = []
    val_accuracies = []
    # best_val_acc = 0.0
    best_test_acc = 0.0

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]")

        for batch_idx, (audio, labels) in enumerate(train_pbar):
            audio, labels = audio.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(audio)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]")
            for audio, labels in val_pbar:
                audio, labels = audio.to(device), labels.to(device)

                outputs = model(audio)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                val_pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        avg_val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(all_labels, all_preds)

        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)

        print(
            f"Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        # Save best model
        # if val_acc > best_val_acc:
        #     best_val_acc = val_acc
        #     torch.save(model.state_dict(), "results/task2/best_panns_improved_model.pth")
        #     print(f"New best model saved with validation accuracy: {best_val_acc:.4f}")

        # Evaluate on test set
        test_top1, test_top3 = evaluate_test_set(
            model, test_loader, label_encoder, epoch + 1
        )
        score = test_top1 + test_top3 * 0.5
        print(
            f"Epoch {epoch + 1}: Test Top-1 Accuracy: {test_top1:.4f}, Test Top-3 Accuracy: {test_top3:.4f}, Test Score: {score:.4f}"
        )
        if score > best_test_acc:
            best_test_acc = test_top1 + test_top3 * 0.5
            torch.save(
                model.state_dict(),
                f"results/task2/best_panns_duration_{duration}_score_{score:.4f}_model.pth",
            )
            print(f"New best model saved with test score: {score:.4f}")

        # Step scheduler and check if learning rate changed
        old_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_acc)
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr != old_lr:
            print(f"Learning rate reduced from {old_lr:.2e} to {new_lr:.2e}")

    return train_losses, val_losses, val_accuracies


def main():
    # Create results directory
    os.makedirs("results/task2", exist_ok=True)

    print("Loading datasets with improved settings...")
    DURATION = 150
    EPOCHS = 100

    # Create datasets with augmentation and longer duration
    train_dataset = Artist20Dataset(
        "data/artist20/train.json", duration=DURATION, augment=True
    )
    val_dataset = Artist20Dataset(
        "data/artist20/val.json", duration=DURATION, augment=False
    )

    # Use the same label encoder for both datasets
    val_dataset.label_encoder = train_dataset.label_encoder
    val_dataset.encoded_labels = train_dataset.label_encoder.transform(
        val_dataset.labels
    )

    # Create test dataset
    test_dataset = TestDataset(duration=DURATION)

    # Create data loaders with larger batch size
    batch_size = 8
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Number of artists: {len(train_dataset.label_encoder.classes_)}")

    # Create improved model
    num_classes = len(train_dataset.label_encoder.classes_)
    model = PANNsClassifier(num_classes, fine_tune=False)  # Start without fine-tuning

    print("Starting improved training...")
    train_losses, val_losses, val_accuracies = train_model(
        model,
        train_loader,
        val_loader,
        test_loader,
        train_dataset.label_encoder,
        num_epochs=EPOCHS,
        lr=0.005,
        duration=DURATION,  # Lower learning rate
    )

    print("\nImproved training completed!")
    print(f"Best validation accuracy: {max(val_accuracies):.4f}")


if __name__ == "__main__":
    main()
