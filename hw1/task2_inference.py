import torch
import torch.nn as nn
import numpy as np
import json
import librosa
import os
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import warnings
from panns_inference import AudioTagging

warnings.filterwarnings("ignore")


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
        top_probs, top_indices = torch.topk(probabilities, k=top_k, dim=1)

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
    # Configuration
    model_path = "assets/best_panns_model.pth"
    train_json_path = "data/artist20/train.json"
    test_dir = "data/artist20/test"
    output_path = "results/task2/test_predictions.json"
    DURATION = 150

    # Create results directory if it doesn't exist
    os.makedirs("results/task2", exist_ok=True)

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please make sure you have trained the model first using task2_panns.py")
        return

    # Check if test directory exists
    if not os.path.exists(test_dir):
        print(f"Error: Test directory not found at {test_dir}")
        print("Please make sure the dataset is properly downloaded and extracted")
        return

    print("Starting inference on test data...")

    # Run inference
    predictions = inference_on_test_data(
        model_path=model_path,
        train_json_path=train_json_path,
        test_dir=test_dir,
        output_path=output_path,
        duration=DURATION,  # Use same duration as training
    )

    print("\nInference completed!")
    print(f"Predictions saved to: {output_path}")

    # Show sample predictions
    print("\nSample predictions:")
    for i, (file_id, preds) in enumerate(list(predictions.items())[:5]):
        print(f"{file_id}: {preds}")


if __name__ == "__main__":
    main()
