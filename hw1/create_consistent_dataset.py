#!/usr/bin/env python3
"""
Create a dataset with consistent feature extraction (same as regular pipeline)
No augmentation to avoid complexity - just use the regular preprocessing with the same features
"""

import json
import os
import numpy as np
from task1_preprocessing import AudioFeatureExtractor


def main():
    """Create dataset with consistent feature extraction"""
    print("🎵 Creating Consistent Dataset for Task 1")
    print("=" * 50)

    # Check if data exists
    if not os.path.exists("data/artist20/train.json"):
        print("❌ Dataset not found. Please run bash get_dataset.sh first.")
        return

    # Load data splits
    with open("data/artist20/train.json", "r") as f:
        train_data = json.load(f)

    with open("data/artist20/val.json", "r") as f:
        val_data = json.load(f)

    # Convert to full paths
    train_files = [f"data/artist20/{path}" for path in train_data]
    val_files = [f"data/artist20/{path}" for path in val_data]

    print(f"Training files: {len(train_files)}")
    print(f"Validation files: {len(val_files)}")

    # Create output directory
    output_dir = "results/task1_augmented_fixed"
    os.makedirs(output_dir, exist_ok=True)

    # Initialize feature extractor - SAME AS REGULAR PIPELINE
    extractor = AudioFeatureExtractor(sr=16000)

    try:
        # Extract training features
        print("Extracting training features...")
        X_train, y_train = extractor.extract_features_batch(train_files, batch_size=8)

        # Extract validation features
        print("Extracting validation features...")
        X_val, y_val = extractor.extract_features_batch(val_files, batch_size=8)

        # Save to files
        np.save(f"{output_dir}/X_train.npy", X_train)
        np.save(f"{output_dir}/y_train.npy", y_train)
        np.save(f"{output_dir}/X_val.npy", X_val)
        np.save(f"{output_dir}/y_val.npy", y_val)

        # Create label mapping
        unique_labels = sorted(set(list(y_train) + list(y_val)))
        label_mapping = {i: label for i, label in enumerate(unique_labels)}

        with open(f"{output_dir}/label_mapping.json", "w") as f:
            json.dump(label_mapping, f, indent=2)

        print("\n✅ Consistent dataset created successfully!")
        print(f"📁 Saved to: {output_dir}/")
        print(f"📊 Dataset statistics:")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Validation samples: {len(X_val)}")
        print(f"   Features per sample: {X_train.shape[1]}")
        print(f"   Number of classes: {len(unique_labels)}")

        print("\n🔧 This uses EXACTLY the same feature extraction as the regular pipeline")
        print("   to ensure consistency between training and test feature extraction.")

    except Exception as e:
        print(f"❌ Error creating dataset: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()