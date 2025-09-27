import numpy as np
import json
import os


def check_preprocessing_results():
    """Check and validate the augmented preprocessing results"""

    results_dir = "results/task1_augmented"

    print("=== Augmented Preprocessing Results Validation ===")
    print(f"Checking results in: {results_dir}")
    print("=" * 50)

    # Check if files exist
    files_to_check = [
        "X_train.npy",
        "X_val.npy",
        "y_train.npy",
        "y_val.npy",
        "label_mapping.json",
        "best_model.pkl",
        "robust_scaler.pkl",
        "standard_scaler.pkl",
        "minmax_scaler.pkl",
        "l2_normalizer.pkl",
        "variance_selector.pkl",
        "feature_selector.pkl",
        "label_encoder.pkl",
        "test_predictions.json",
    ]

    # Optional files
    optional_files = [
        "power_transformer.pkl",
        "reports/classification_report.txt",
        "reports/results_summary.json",
    ]

    missing_files = []
    for file in files_to_check:
        file_path = os.path.join(results_dir, file)
        if os.path.exists(file_path):
            print(f"✅ {file} - Found")
        else:
            print(f"❌ {file} - Missing")
            missing_files.append(file)

    print("\nOptional files:")
    for file in optional_files:
        file_path = os.path.join(results_dir, file)
        if os.path.exists(file_path):
            print(f"✅ {file} - Found")
        else:
            print(f"⚠️  {file} - Optional (not found)")

    if missing_files:
        print(f"\n❌ Critical files missing: {missing_files}")
        print("Please run the augmented training pipeline first.")
        return

    print("\n1. LOADING DATA")
    print("-" * 20)

    try:
        # Load all data
        X_train = np.load(os.path.join(results_dir, "X_train.npy"))
        X_val = np.load(os.path.join(results_dir, "X_val.npy"))
        y_train = np.load(os.path.join(results_dir, "y_train.npy"))
        y_val = np.load(os.path.join(results_dir, "y_val.npy"))

        with open(os.path.join(results_dir, "label_mapping.json"), "r") as f:
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
    print("Expected artists:             20")

    print("\nArtist distribution in training:")
    train_counts = {}
    for artist in unique_train_artists:
        count = np.sum(y_train == artist)
        train_counts[artist] = count
        print(f"  {artist}: {count} samples")

    print("\nArtist distribution in validation:")
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

    print("Training features:")
    print(f"  NaN values: {train_nan}")
    print(f"  Inf values: {train_inf}")
    print(f"  Min value:  {X_train.min():.6f}")
    print(f"  Max value:  {X_train.max():.6f}")
    print(f"  Mean:       {X_train.mean():.6f}")
    print(f"  Std:        {X_train.std():.6f}")

    print("\nValidation features:")
    print(f"  NaN values: {val_nan}")
    print(f"  Inf values: {val_inf}")
    print(f"  Min value:  {X_val.min():.6f}")
    print(f"  Max value:  {X_val.max():.6f}")
    print(f"  Mean:       {X_val.mean():.6f}")
    print(f"  Std:        {X_val.std():.6f}")

    # Show sample features
    print("\n6. SAMPLE FEATURES")
    print("-" * 18)
    print("First training sample features (first 10):")
    print(f"  {X_train[0][:10]}")
    print(f"  Artist: {y_train[0]}")

    print("\nFirst validation sample features (first 10):")
    print(f"  {X_val[0][:10]}")
    print(f"  Artist: {y_val[0]}")

    # Check for zero features
    zero_features_train = np.sum(X_train == 0, axis=0)
    zero_features_val = np.sum(X_val == 0, axis=0)

    print("\nFeatures with all zeros:")
    print(f"  Training: {np.sum(zero_features_train == X_train.shape[0])} features")
    print(f"  Validation: {np.sum(zero_features_val == X_val.shape[0])} features")

    print("\n7. VALIDATION SUMMARY")
    print("-" * 21)

    # Summary checks
    issues = []

    if len(all_artists) != 20:
        issues.append(f"Expected 20 artists, found {len(all_artists)}")

    # For augmented data, we expect more samples due to augmentation
    if X_train.shape[0] < 949:
        issues.append(f"Expected at least 949 training samples (before augmentation), found {X_train.shape[0]}")

    if X_val.shape[0] < 231:
        issues.append(f"Expected at least 231 validation samples (before augmentation), found {X_val.shape[0]}")

    if train_nan > 0 or val_nan > 0:
        issues.append("Found NaN values in features")

    if train_inf > 0 or val_inf > 0:
        issues.append("Found infinite values in features")

    # Additional augmented data checks
    print("\n8. AUGMENTED DATA ANALYSIS")
    print("-" * 27)

    # Check if we have test predictions
    test_pred_path = os.path.join(results_dir, "test_predictions.json")
    if os.path.exists(test_pred_path):
        try:
            with open(test_pred_path, "r") as f:
                test_predictions = json.load(f)
            print(f"✅ Test predictions found: {len(test_predictions)} predictions")

            # Check format
            sample_key = list(test_predictions.keys())[0]
            sample_pred = test_predictions[sample_key]
            if isinstance(sample_pred, list) and len(sample_pred) == 3:
                print("✅ Test predictions format: Correct (top-3 predictions per file)")
                print(f"   Sample prediction for {sample_key}: {sample_pred}")
            else:
                issues.append("Test predictions format incorrect (should be top-3 list)")
        except Exception as e:
            issues.append(f"Error reading test predictions: {e}")
    else:
        print("⚠️  Test predictions not found")

    # Check augmentation factor
    augmentation_factor = X_train.shape[0] / 949  # Original training samples
    print(f"Augmentation factor: {augmentation_factor:.2f}x (Training: {X_train.shape[0]} vs original 949)")

    validation_factor = X_val.shape[0] / 231  # Original validation samples
    print(f"Validation factor: {validation_factor:.2f}x (Validation: {X_val.shape[0]} vs original 231)")

    # Check model files
    model_files = ["best_model.pkl", "robust_scaler.pkl", "standard_scaler.pkl",
                   "minmax_scaler.pkl", "l2_normalizer.pkl", "variance_selector.pkl",
                   "feature_selector.pkl", "label_encoder.pkl"]

    print("\nModel and preprocessor files:")
    for model_file in model_files:
        model_path = os.path.join(results_dir, model_file)
        if os.path.exists(model_path):
            file_size = os.path.getsize(model_path) / 1024  # KB
            print(f"  ✅ {model_file}: {file_size:.1f} KB")
        else:
            print(f"  ❌ {model_file}: Missing")

    if len(issues) == 0:
        print("\n✅ ALL CHECKS PASSED!")
        print("   - Correct number of artists (20)")
        print(f"   - Augmented training samples: {X_train.shape[0]} ({augmentation_factor:.1f}x)")
        print(f"   - Augmented validation samples: {X_val.shape[0]} ({validation_factor:.1f}x)")
        print("   - No NaN or infinite values")
        print("   - Features extracted successfully")
        print("   - Model files saved")
        if os.path.exists(test_pred_path):
            print("   - Test predictions generated")
        print("\n🚀 Augmented data pipeline completed successfully!")
    else:
        print("❌ ISSUES FOUND:")
        for issue in issues:
            print(f"   - {issue}")

    print(f"\nFiles saved in: {os.path.abspath(results_dir)}/")

    # Show directory structure
    print("\n📁 Directory structure:")
    for root, dirs, files in os.walk(results_dir):
        level = root.replace(results_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files[:10]:  # Show first 10 files to avoid clutter
            file_size = os.path.getsize(os.path.join(root, file)) / 1024
            print(f"{subindent}{file} ({file_size:.1f} KB)")
        if len(files) > 10:
            print(f"{subindent}... and {len(files) - 10} more files")


def check_augmentation_quality():
    """Additional checks specific to augmented data quality"""

    results_dir = "results/task1_augmented"

    print("\n" + "=" * 60)
    print("🔍 AUGMENTED DATA QUALITY ANALYSIS")
    print("=" * 60)

    try:
        X_train = np.load(os.path.join(results_dir, "X_train.npy"))
        y_train = np.load(os.path.join(results_dir, "y_train.npy"))

        print(f"\n1. FEATURE STATISTICS")
        print("-" * 22)
        print(f"Feature dimensionality: {X_train.shape[1]}")
        print(f"Feature range: [{X_train.min():.6f}, {X_train.max():.6f}]")
        print(f"Feature mean: {X_train.mean():.6f}")
        print(f"Feature std: {X_train.std():.6f}")

        # Check feature distribution
        zero_features = np.sum(X_train == 0, axis=0)
        constant_features = np.sum(zero_features == X_train.shape[0])

        print(f"\n2. FEATURE QUALITY")
        print("-" * 17)
        print(f"Constant features: {constant_features}")
        print(f"Near-zero variance features: {np.sum(np.var(X_train, axis=0) < 1e-6)}")

        # Check for potential data leakage or artifacts
        duplicate_samples = 0
        for i in range(min(1000, X_train.shape[0])):
            for j in range(i+1, min(1000, X_train.shape[0])):
                if np.allclose(X_train[i], X_train[j], rtol=1e-10):
                    duplicate_samples += 1
                    break

        print(f"Potential duplicate samples (first 1000 checked): {duplicate_samples}")

        # Artist distribution analysis
        print(f"\n3. AUGMENTATION BALANCE")
        print("-" * 24)
        unique_artists, counts = np.unique(y_train, return_counts=True)
        min_samples = counts.min()
        max_samples = counts.max()
        balance_ratio = min_samples / max_samples

        print(f"Artist balance ratio: {balance_ratio:.3f} (1.0 = perfect balance)")
        print(f"Samples per artist: {min_samples} - {max_samples}")

        if balance_ratio < 0.5:
            print("⚠️  Warning: Significant class imbalance detected")
        else:
            print("✅ Good class balance")

    except Exception as e:
        print(f"❌ Error analyzing augmented data: {e}")


if __name__ == "__main__":
    check_preprocessing_results()
    check_augmentation_quality()
