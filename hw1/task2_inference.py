import torch
import torch.nn as nn
import numpy as np
import json
import librosa
import os
import argparse
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import warnings
import urllib.request
from pathlib import Path

warnings.filterwarnings("ignore")


def download_panns_files():
    """Download all required PANNs files using cross-platform approach"""
    assets_dir = "assets"
    os.makedirs(assets_dir, exist_ok=True)

    # Download checkpoint
    checkpoint_path = os.path.join(assets_dir, "Cnn14_mAP=0.431.pth")
    if not os.path.exists(checkpoint_path) or os.path.getsize(checkpoint_path) < 3e8:
        print(f"Downloading PANNs checkpoint to: {checkpoint_path}")
        print("This may take a few minutes...")
        try:
            url = "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"
            urllib.request.urlretrieve(url, checkpoint_path)
            print("Checkpoint download completed!")
        except Exception as e:
            print(f"Error downloading checkpoint: {e}")
            raise

    # Download class labels CSV
    labels_csv_path = os.path.join(assets_dir, "class_labels_indices.csv")
    if not os.path.exists(labels_csv_path):
        print(f"Downloading class labels to: {labels_csv_path}")
        try:
            url = "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv"
            urllib.request.urlretrieve(url, labels_csv_path)
            print("Labels download completed!")
        except Exception as e:
            print(f"Error downloading labels: {e}")
            raise

    return checkpoint_path, labels_csv_path


def setup_panns_environment():
    """Set up PANNs environment with downloaded files"""
    checkpoint_path, labels_csv_path = download_panns_files()

    # Create symbolic links or copy files to expected locations if needed
    home_panns_dir = os.path.join(str(Path.home()), "panns_data")
    os.makedirs(home_panns_dir, exist_ok=True)

    home_checkpoint = os.path.join(home_panns_dir, "Cnn14_mAP=0.431.pth")
    home_labels = os.path.join(home_panns_dir, "class_labels_indices.csv")

    # Copy files if they don't exist in home directory
    if not os.path.exists(home_checkpoint):
        import shutil

        shutil.copy2(checkpoint_path, home_checkpoint)
        print(f"Copied checkpoint to {home_checkpoint}")

    if not os.path.exists(home_labels):
        import shutil

        shutil.copy2(labels_csv_path, home_labels)
        print(f"Copied labels to {home_labels}")

    return checkpoint_path


class PANNsClassifier(nn.Module):
    def __init__(self, num_classes, feature_dim=2048, fine_tune=False):
        super().__init__()

        # Set up PANNs environment and download required files
        checkpoint_path = setup_panns_environment()

        # Import AudioTagging after setting up the environment
        from panns_inference import AudioTagging

        self.panns = AudioTagging(
            checkpoint_path=checkpoint_path,
            device="cuda" if torch.cuda.is_available() else "cpu",
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


def load_model_and_encoder(model_path, train_json_path):
    """Load the trained model and recreate the label encoder from training data"""

    # Recreate label encoder from training data
    with open(train_json_path, "r") as f:
        train_file_paths = json.load(f)

    # Extract labels from file paths (artist names)
    train_labels = []
    for file_path in train_file_paths:
        path_parts = file_path.split("/")
        artist_name = path_parts[2]
        train_labels.append(artist_name)

    label_encoder = LabelEncoder()
    label_encoder.fit(train_labels)

    print(f"Label encoder created with {len(label_encoder.classes_)} classes:")
    print(f"Artists: {list(label_encoder.classes_)}")

    # Create model
    num_classes = len(label_encoder.classes_)
    model = PANNsClassifier(num_classes, fine_tune=False)

    # Load trained weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    print(f"Model loaded successfully on {device}")

    return model, label_encoder


def preprocess_audio(audio_path, sr=16000, duration=120):
    """Preprocess audio file for inference"""
    try:
        # Load audio
        audio, _ = librosa.load(audio_path, sr=sr)

        # Calculate required samples
        samples = int(sr * duration)

        # Handle audio length
        if len(audio) > samples:
            # Take from beginning (no random crop for inference)
            audio = audio[:samples]
        else:
            # Pad if shorter
            audio = np.pad(audio, (0, max(0, samples - len(audio))), mode="constant")

        return torch.FloatTensor(audio).unsqueeze(0)  # Add batch dimension

    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        return torch.zeros(1, samples)


def predict_single_file(model, audio_tensor, label_encoder, device, top_k=3):
    """Make prediction for a single audio file"""
    audio_tensor = audio_tensor.to(device)

    with torch.no_grad():
        outputs = model(audio_tensor)
        probabilities = torch.softmax(outputs, dim=1)

        # Get top-k predictions
        _, top_indices = torch.topk(probabilities, k=top_k, dim=1)

        # Convert to artist names
        predictions = []
        for idx in top_indices[0]:
            artist_name = label_encoder.inverse_transform([idx.cpu().numpy()])[0]
            predictions.append(artist_name)

    return predictions


def inference_on_test_data(
    model_path, train_json_path, test_dir, output_path, duration=120
):
    """Run inference on all test files and save predictions"""

    # Load model and label encoder
    model, label_encoder = load_model_and_encoder(model_path, train_json_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get all test files
    test_files = sorted([f for f in os.listdir(test_dir) if f.endswith(".mp3")])
    print(f"Found {len(test_files)} test files")

    # Dictionary to store predictions
    predictions = {}

    # Process each test file
    for filename in tqdm(test_files, desc="Processing test files"):
        file_path = os.path.join(test_dir, filename)
        file_id = filename.replace(".mp3", "")  # Remove extension for ID

        # Preprocess audio
        audio_tensor = preprocess_audio(file_path, duration=duration)

        # Make prediction
        top3_predictions = predict_single_file(
            model, audio_tensor, label_encoder, device, top_k=3
        )

        # Store prediction
        predictions[file_id] = top3_predictions

        # Print first few predictions for verification
        if len(predictions) <= 5:
            print(f"{file_id}: {top3_predictions}")

    # Save predictions to JSON
    with open(output_path, "w") as f:
        json.dump(predictions, f, indent=2)

    print(f"\nPredictions saved to: {output_path}")
    print(f"Total predictions: {len(predictions)}")

    return predictions


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run inference on test data using trained PANNs model"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="assets/best_panns_model.pth",
        help="Path to trained model file (default: assets/best_panns_model.pth)",
    )
    parser.add_argument(
        "--train_json",
        type=str,
        default="data/artist20/train.json",
        help="Path to training JSON for label encoder (default: data/artist20/train.json)",
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default="data/artist20/test",
        help="Path to test data directory (default: data/artist20/test)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="r14921046.json",
        help="Path to output predictions file (default: r14921046.json)",
    )

    args = parser.parse_args()

    # Fixed duration matching training configuration
    DURATION = 150

    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        print("Please make sure you have trained the model first using task2_train.py")
        return

    # Check if test directory exists
    if not os.path.exists(args.test_dir):
        print(f"Error: Test directory not found at {args.test_dir}")
        print("Please make sure the dataset is properly downloaded and extracted")
        return

    print("Starting inference on test data...")
    print(f"Model: {args.model_path}")
    print(f"Test directory: {args.test_dir}")
    print(f"Output: {args.output}")
    print(f"Audio duration: {DURATION}s")

    # Run inference
    predictions = inference_on_test_data(
        model_path=args.model_path,
        train_json_path=args.train_json,
        test_dir=args.test_dir,
        output_path=args.output,
        duration=DURATION,
    )

    print("\nInference completed!")
    print(f"Predictions saved to: {args.output}")

    # Show sample predictions
    print("\nSample predictions:")
    for file_id, preds in list(predictions.items())[:5]:
        print(f"{file_id}: {preds}")


if __name__ == "__main__":
    main()
