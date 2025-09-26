import numpy as np
import json
import os

def check_preprocessing_results():
    """Check and validate the preprocessing results"""

    results_dir = "results/task1"

    print("=== Preprocessing Results Validation ===")
    print(f"Checking results in: {results_dir}")
    print("=" * 50)

    # Check if files exist
    files_to_check = ["X_train.npy", "X_val.npy", "y_train.npy", "y_val.npy", "label_mapping.json"]

    for file in files_to_check:
        file_path = os.path.join(results_dir, file)
        if os.path.exists(file_path):
            print(f"✅ {file} - Found")
        else:
            print(f"❌ {file} - Missing")
            return

    print("\n1. LOADING DATA")
    print("-" * 20)

    try:
        # Load all data
        X_train = np.load(os.path.join(results_dir, "X_train.npy"))
        X_val = np.load(os.path.join(results_dir, "X_val.npy"))
        y_train = np.load(os.path.join(results_dir, "y_train.npy"))
        y_val = np.load(os.path.join(results_dir, "y_val.npy"))

        with open(os.path.join(results_dir, "label_mapping.json"), 'r') as f:
            label_mapping = json.load(f)

        print("All data loaded successfully!")

    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print("\n2. DATA SHAPES & SIZES")
    print("-" * 25)
    print(f"Training features (X_train): {X_train.shape}")
    print(f"Training labels (y_train):   {y_train.shape}")
    print(f"Validation features (X_val): {X_val.shape}")
    print(f"Validation labels (y_val):   {y_val.shape}")
    print(f"Feature dimensions:          {X_train.shape[1]} features per sample")

    print("\n3. LABEL ANALYSIS")
    print("-" * 18)

    # Check unique artists
    unique_train_artists = np.unique(y_train)
    unique_val_artists = np.unique(y_val)
    all_artists = np.unique(np.concatenate([y_train, y_val]))

    print(f"Unique artists in training:   {len(unique_train_artists)}")
    print(f"Unique artists in validation: {len(unique_val_artists)}")
    print(f"Total unique artists:         {len(all_artists)}")
    print(f"Expected artists:             20")

    print(f"\nArtist distribution in training:")
    train_counts = {}
    for artist in unique_train_artists:
        count = np.sum(y_train == artist)
        train_counts[artist] = count
        print(f"  {artist}: {count} samples")

    print(f"\nArtist distribution in validation:")
    val_counts = {}
    for artist in unique_val_artists:
        count = np.sum(y_val == artist)
        val_counts[artist] = count
        print(f"  {artist}: {count} samples")

    print("\n4. LABEL MAPPING")
    print("-" * 17)
    print(f"Artists in label mapping: {len(label_mapping)}")
    print("Label mapping:")
    for artist, idx in sorted(label_mapping.items(), key=lambda x: x[1]):
        print(f"  {idx}: {artist}")

    print("\n5. FEATURE QUALITY CHECK")
    print("-" * 24)

    # Check for NaN or infinite values
    train_nan = np.sum(np.isnan(X_train))
    train_inf = np.sum(np.isinf(X_train))
    val_nan = np.sum(np.isnan(X_val))
    val_inf = np.sum(np.isinf(X_val))

    print(f"Training features:")
    print(f"  NaN values: {train_nan}")
    print(f"  Inf values: {train_inf}")
    print(f"  Min value:  {X_train.min():.6f}")
    print(f"  Max value:  {X_train.max():.6f}")
    print(f"  Mean:       {X_train.mean():.6f}")
    print(f"  Std:        {X_train.std():.6f}")

    print(f"\nValidation features:")
    print(f"  NaN values: {val_nan}")
    print(f"  Inf values: {val_inf}")
    print(f"  Min value:  {X_val.min():.6f}")
    print(f"  Max value:  {X_val.max():.6f}")
    print(f"  Mean:       {X_val.mean():.6f}")
    print(f"  Std:        {X_val.std():.6f}")

    # Show sample features
    print("\n6. SAMPLE FEATURES")
    print("-" * 18)
    print(f"First training sample features (first 10):")
    print(f"  {X_train[0][:10]}")
    print(f"  Artist: {y_train[0]}")

    print(f"\nFirst validation sample features (first 10):")
    print(f"  {X_val[0][:10]}")
    print(f"  Artist: {y_val[0]}")

    # Check for zero features
    zero_features_train = np.sum(X_train == 0, axis=0)
    zero_features_val = np.sum(X_val == 0, axis=0)

    print(f"\nFeatures with all zeros:")
    print(f"  Training: {np.sum(zero_features_train == X_train.shape[0])} features")
    print(f"  Validation: {np.sum(zero_features_val == X_val.shape[0])} features")

    print("\n7. VALIDATION SUMMARY")
    print("-" * 21)

    # Summary checks
    issues = []

    if len(all_artists) != 20:
        issues.append(f"Expected 20 artists, found {len(all_artists)}")

    if X_train.shape[0] != 949:
        issues.append(f"Expected 949 training samples, found {X_train.shape[0]}")

    if X_val.shape[0] != 231:
        issues.append(f"Expected 231 validation samples, found {X_val.shape[0]}")

    if train_nan > 0 or val_nan > 0:
        issues.append("Found NaN values in features")

    if train_inf > 0 or val_inf > 0:
        issues.append("Found infinite values in features")

    if len(issues) == 0:
        print("✅ ALL CHECKS PASSED!")
        print("   - Correct number of artists (20)")
        print("   - Correct number of samples (949 train, 231 val)")
        print("   - No NaN or infinite values")
        print("   - Features extracted successfully")
        print("\n🚀 Data is ready for model training!")
    else:
        print("❌ ISSUES FOUND:")
        for issue in issues:
            print(f"   - {issue}")

    print(f"\nFiles saved in: {os.path.abspath(results_dir)}/")

if __name__ == "__main__":
    check_preprocessing_results()