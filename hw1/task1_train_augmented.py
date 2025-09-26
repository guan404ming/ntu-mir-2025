import numpy as np
import json
import pickle
import os
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch
import warnings

# GPU-accelerated ML libraries
try:
    import cuml
    from cuml.svm import SVC as cuSVC
    from cuml.ensemble import RandomForestClassifier as cuRandomForest
    from cuml.neighbors import KNeighborsClassifier as cuKNN

    GPU_AVAILABLE = True
    print("🚀 GPU acceleration available with cuML")
except ImportError:
    GPU_AVAILABLE = False
    print("⚠️  cuML not available, using CPU-only models")

try:
    import xgboost as xgb

    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb

    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

warnings.filterwarnings("ignore")


class AugmentedTraditionalMLPipeline:
    """
    Traditional ML Pipeline optimized for augmented audio data
    """

    def __init__(self, use_augmented_data=True):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_augmented_data = use_augmented_data
        self.data_dir = (
            "results/task1_augmented" if use_augmented_data else "results/task1"
        )

        # Enhanced preprocessing for augmented data
        self.scaler = RobustScaler()  # Better for augmented data with outliers
        self.variance_selector = VarianceThreshold(
            threshold=0.01
        )  # Remove low-variance features
        self.feature_selector = SelectKBest(f_classif, k="all")  # Will be tuned
        self.label_encoder = LabelEncoder()

        self.models = {}
        self.best_model = None
        self.class_weights = None

        print("Augmented Traditional ML Pipeline initialized")
        print(f"Using data from: {self.data_dir}")
        print(f"GPU available: {torch.cuda.is_available()}")

    def load_data(self):
        """Load preprocessed features"""
        print("Loading preprocessed features...")

        if not os.path.exists(f"{self.data_dir}/X_train.npy"):
            raise FileNotFoundError(
                f"Features not found in {self.data_dir}/. Run preprocessing first."
            )

        X_train = np.load(f"{self.data_dir}/X_train.npy")
        y_train = np.load(f"{self.data_dir}/y_train.npy")
        X_val = np.load(f"{self.data_dir}/X_val.npy")
        y_val = np.load(f"{self.data_dir}/y_val.npy")

        with open(f"{self.data_dir}/label_mapping.json", "r") as f:
            label_mapping = json.load(f)

        print(f"Training data: {X_train.shape}")
        print(f"Validation data: {X_val.shape}")
        print(f"Number of classes: {len(label_mapping)}")

        # Check for class imbalance
        unique, counts = np.unique(y_train, return_counts=True)
        print("Class distribution in training set:")
        for label, count in zip(unique, counts):
            print(f"  {label}: {count} samples")

        return X_train, y_train, X_val, y_val, label_mapping

    def preprocess_data(self, X_train, y_train, X_val, y_val):
        """Enhanced preprocessing for augmented data"""
        print("Preprocessing augmented data...")

        # Handle NaN values more aggressively
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e6, neginf=-1e6)
        X_val = np.nan_to_num(X_val, nan=0.0, posinf=1e6, neginf=-1e6)

        # Remove constant features
        print("Removing low-variance features...")
        X_train = self.variance_selector.fit_transform(X_train)
        X_val = self.variance_selector.transform(X_val)
        print(f"Features after variance filtering: {X_train.shape[1]}")

        # Robust scaling
        print("Applying robust scaling...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # Feature selection with cross-validation
        print("Selecting best features...")
        # Determine optimal number of features
        n_features_options = [50, 100, 200, min(500, X_train_scaled.shape[1])]
        best_score = 0
        best_k = 100

        for k in n_features_options:
            if k >= X_train_scaled.shape[1]:
                k = X_train_scaled.shape[1]

            selector = SelectKBest(f_classif, k=k)
            X_temp = selector.fit_transform(X_train_scaled, y_train)

            # Quick SVM test to evaluate feature set
            svm_quick = SVC(kernel="rbf", C=1.0, random_state=42)
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            scores = []

            for train_idx, val_idx in cv.split(X_temp, y_train):
                svm_quick.fit(X_temp[train_idx], y_train[train_idx])
                score = svm_quick.score(X_temp[val_idx], y_train[val_idx])
                scores.append(score)

            avg_score = np.mean(scores)
            print(f"  k={k}: CV score = {avg_score:.4f}")

            if avg_score > best_score:
                best_score = avg_score
                best_k = k

        print(f"Selected {best_k} features (CV score: {best_score:.4f})")
        self.feature_selector.k = best_k
        X_train_selected = self.feature_selector.fit_transform(X_train_scaled, y_train)
        X_val_selected = self.feature_selector.transform(X_val_scaled)

        # Encode labels - fit on all unique labels from both train and val
        all_labels = np.concatenate([y_train, y_val])
        unique_labels = np.unique(all_labels)
        self.label_encoder.fit(unique_labels)

        y_train_encoded = self.label_encoder.transform(y_train)
        y_val_encoded = self.label_encoder.transform(y_val)

        # Compute class weights to handle imbalance
        # Only compute weights for classes that actually appear in training
        train_classes = np.unique(y_train_encoded)
        if len(train_classes) < len(self.label_encoder.classes_):
            # Handle case where some classes don't appear in training
            weights = np.ones(len(self.label_encoder.classes_))
            train_weights = compute_class_weight(
                "balanced", classes=train_classes, y=y_train_encoded
            )
            for i, class_idx in enumerate(train_classes):
                weights[class_idx] = train_weights[i]
            self.class_weights = weights
        else:
            self.class_weights = compute_class_weight(
                "balanced", classes=train_classes, y=y_train_encoded
            )

        class_weight_dict = {i: weight for i, weight in enumerate(self.class_weights)}
        print("Computed class weights for imbalanced data")

        print("Final preprocessing completed.")
        print(f"  Features: {X_train_selected.shape[1]}")
        print(f"  Training samples: {X_train_selected.shape[0]}")
        print(f"  Validation samples: {X_val_selected.shape[0]}")

        return (
            X_train_selected,
            y_train_encoded,
            X_val_selected,
            y_val_encoded,
            class_weight_dict,
        )

    def train_models(self, X_train, y_train, class_weight_dict):
        """Train models optimized for augmented data with GPU acceleration"""
        print("Training models on augmented dataset...")

        # Convert to CuPy arrays if GPU is available
        if GPU_AVAILABLE:
            try:
                import cupy as cp

                X_train_gpu = cp.asarray(X_train)
                y_train_gpu = cp.asarray(y_train)
                print("🚀 Using GPU arrays for training")
            except ImportError:
                X_train_gpu = X_train
                y_train_gpu = y_train
                print("⚠️  CuPy not available, using CPU arrays")
        else:
            X_train_gpu = X_train
            y_train_gpu = y_train

        # Define models with GPU acceleration when available
        models_to_train = []

        if GPU_AVAILABLE:
            models_to_train.extend(
                [
                    ("cuSVM_RBF", "GPU-accelerated SVM with RBF kernel"),
                    ("cuRandomForest", "GPU-accelerated Random Forest"),
                    ("cuKNN", "GPU-accelerated k-Nearest Neighbors"),
                ]
            )

        if XGB_AVAILABLE:
            models_to_train.append(("XGBoost", "XGBoost with GPU acceleration"))

        if LGB_AVAILABLE:
            models_to_train.append(("LightGBM", "LightGBM with GPU acceleration"))

        # Fallback to CPU models
        models_to_train.extend(
            [
                ("SVM_RBF", "Support Vector Machine with RBF kernel (CPU)"),
                ("RandomForest", "Random Forest with class balancing (CPU)"),
            ]
        )

        with tqdm(
            total=len(models_to_train),
            desc="Training models",
            unit="model",
            colour="blue",
        ) as pbar:
            # GPU-accelerated models
            if GPU_AVAILABLE:
                # 1. cuML SVM with RBF kernel
                pbar.set_description("Training cuSVM (RBF)")
                try:
                    cu_svm_rbf = cuSVC(
                        C=10.0,
                        gamma="scale",
                        kernel="rbf",
                        probability=True,
                        cache_size=1000.0,
                    )
                    cu_svm_rbf.fit(X_train_gpu, y_train_gpu)
                    self.models["cuSVM_RBF"] = cu_svm_rbf
                    print("✅ GPU SVM trained successfully")
                except Exception as e:
                    print(f"❌ GPU SVM failed: {e}")
                pbar.update(1)

                # 2. cuML Random Forest
                pbar.set_description("Training cuRandomForest")
                try:
                    cu_rf = cuRandomForest(
                        n_estimators=100,
                        max_depth=15,
                        max_features="sqrt",
                        random_state=42,
                    )
                    cu_rf.fit(X_train_gpu, y_train_gpu)
                    self.models["cuRandomForest"] = cu_rf
                    print("✅ GPU Random Forest trained successfully")
                except Exception as e:
                    print(f"❌ GPU Random Forest failed: {e}")
                pbar.update(1)

                # 3. cuML KNN
                pbar.set_description("Training cuKNN")
                try:
                    cu_knn = cuKNN(n_neighbors=5, metric="euclidean")
                    cu_knn.fit(X_train_gpu, y_train_gpu)
                    self.models["cuKNN"] = cu_knn
                    print("✅ GPU KNN trained successfully")
                except Exception as e:
                    print(f"❌ GPU KNN failed: {e}")
                pbar.update(1)

            # XGBoost with GPU
            if XGB_AVAILABLE:
                pbar.set_description("Training XGBoost")
                try:
                    # Convert labels to 0-based for XGBoost
                    y_train_xgb = y_train.copy()
                    if hasattr(y_train_xgb, "min") and y_train_xgb.min() > 0:
                        y_train_xgb = y_train_xgb - y_train_xgb.min()

                    xgb_model = xgb.XGBClassifier(
                        tree_method="gpu_hist",
                        gpu_id=0,
                        n_estimators=100,
                        max_depth=6,
                        learning_rate=0.1,
                        random_state=42,
                        eval_metric="mlogloss",
                    )
                    xgb_model.fit(X_train, y_train_xgb)
                    self.models["XGBoost"] = xgb_model
                    print("✅ XGBoost with GPU trained successfully")
                except Exception as e:
                    print(f"❌ XGBoost failed, trying CPU: {e}")
                    try:
                        # Convert labels to 0-based for XGBoost
                        y_train_xgb = y_train.copy()
                        if hasattr(y_train_xgb, "min") and y_train_xgb.min() > 0:
                            y_train_xgb = y_train_xgb - y_train_xgb.min()

                        xgb_model = xgb.XGBClassifier(
                            n_estimators=100,
                            max_depth=6,
                            learning_rate=0.1,
                            random_state=42,
                            eval_metric="mlogloss",
                        )
                        xgb_model.fit(X_train, y_train_xgb)
                        self.models["XGBoost"] = xgb_model
                        print("✅ XGBoost CPU trained successfully")
                    except Exception as e2:
                        print(f"❌ XGBoost completely failed: {e2}")
                pbar.update(1)

            # LightGBM with GPU
            if LGB_AVAILABLE:
                pbar.set_description("Training LightGBM")
                try:
                    # Convert labels to 0-based for LightGBM
                    y_train_lgb = y_train.copy()
                    if hasattr(y_train_lgb, "min") and y_train_lgb.min() > 0:
                        y_train_lgb = y_train_lgb - y_train_lgb.min()

                    lgb_model = lgb.LGBMClassifier(
                        device="gpu",
                        gpu_platform_id=0,
                        gpu_device_id=0,
                        n_estimators=100,
                        max_depth=6,
                        learning_rate=0.1,
                        random_state=42,
                        verbose=-1,
                    )
                    lgb_model.fit(X_train, y_train_lgb)
                    self.models["LightGBM"] = lgb_model
                    print("✅ LightGBM with GPU trained successfully")
                except Exception as e:
                    print(f"❌ LightGBM GPU failed, trying CPU: {e}")
                    try:
                        # Convert labels to 0-based for LightGBM
                        y_train_lgb = y_train.copy()
                        if hasattr(y_train_lgb, "min") and y_train_lgb.min() > 0:
                            y_train_lgb = y_train_lgb - y_train_lgb.min()

                        lgb_model = lgb.LGBMClassifier(
                            n_estimators=100,
                            max_depth=6,
                            learning_rate=0.1,
                            random_state=42,
                            verbose=-1,
                        )
                        lgb_model.fit(X_train, y_train_lgb)
                        self.models["LightGBM"] = lgb_model
                        print("✅ LightGBM CPU trained successfully")
                    except Exception as e2:
                        print(f"❌ LightGBM completely failed: {e2}")
                pbar.update(1)

            # Enhanced CPU models with hyperparameter tuning
            pbar.set_description("Training Enhanced SVM (CPU)")
            # Grid search for SVM hyperparameters
            svm_param_grid = {
                'C': [1.0, 10.0, 100.0],
                'gamma': ['scale', 'auto', 0.001, 0.01],
                'kernel': ['rbf', 'poly']
            }
            svm_base = SVC(probability=True, random_state=42, class_weight="balanced")
            svm_grid = GridSearchCV(svm_base, svm_param_grid, cv=3, scoring='accuracy', n_jobs=-1)
            svm_grid.fit(X_train, y_train)
            self.models["SVM_Enhanced"] = svm_grid
            pbar.update(1)

            pbar.set_description("Training Enhanced RandomForest (CPU)")
            # Grid search for RandomForest hyperparameters
            rf_param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 15, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
            rf_base = RandomForestClassifier(random_state=42, class_weight="balanced", n_jobs=-1)
            rf_grid = GridSearchCV(rf_base, rf_param_grid, cv=3, scoring='accuracy', n_jobs=-1)
            rf_grid.fit(X_train, y_train)
            self.models["RandomForest_Enhanced"] = rf_grid
            pbar.update(1)

            # Add ensemble methods
            pbar.set_description("Training Ensemble Models")
            try:
                from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier

                # Extra Trees
                et = ExtraTreesClassifier(
                    n_estimators=100,
                    max_depth=15,
                    random_state=42,
                    class_weight="balanced",
                    n_jobs=-1
                )
                et.fit(X_train, y_train)
                self.models["ExtraTrees"] = et

                # Gradient Boosting
                gb = GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42
                )
                gb.fit(X_train, y_train)
                self.models["GradientBoosting"] = gb

                # Voting Classifier (ensemble of best models)
                voting_estimators = [
                    ('svm', svm_grid.best_estimator_),
                    ('rf', rf_grid.best_estimator_),
                    ('et', et)
                ]

                voting_clf = VotingClassifier(
                    estimators=voting_estimators,
                    voting='soft',
                    n_jobs=-1
                )
                voting_clf.fit(X_train, y_train)
                self.models["VotingEnsemble"] = voting_clf

                print("✅ Enhanced ensemble models trained successfully")
            except Exception as e:
                print(f"❌ Ensemble training failed: {e}")

            pbar.update(3)  # For the 3 ensemble models
        print(f"✅ All models trained successfully! ({len(self.models)} models)")

        # Print summary
        print("\n📊 Training Results:")
        for name in self.models.keys():
            print(f"  ✅ {name}: Model trained and ready")

        return self.models

    def evaluate_models(self, X_val, y_val):
        """Evaluate all trained models"""
        print("\nEvaluating models on validation set...")

        results = {}
        model_names = list(self.models.keys())

        with tqdm(
            total=len(model_names),
            desc="Evaluating models",
            unit="model",
            colour="green",
        ) as pbar:
            for name, model in self.models.items():
                pbar.set_description(f"Evaluating {name}")

                # Predictions
                y_pred = model.predict(X_val)
                y_pred_proba = model.predict_proba(X_val)

                # Top-1 accuracy
                top1_acc = accuracy_score(y_val, y_pred)

                # Top-3 accuracy
                top3_pred = np.argsort(y_pred_proba, axis=1)[:, -3:][:, ::-1]
                top3_acc = np.mean(
                    [y_val[i] in top3_pred[i] for i in range(len(y_val))]
                )

                # Handle different model types (GridSearchCV vs direct models)
                cv_score = getattr(model, "best_score_", "N/A (direct training)")

                results[name] = {
                    "top1_accuracy": top1_acc,
                    "top3_accuracy": top3_acc,
                    "predictions": y_pred,
                    "probabilities": y_pred_proba,
                    "cv_score": cv_score,
                }

                pbar.set_postfix(
                    {
                        "model": name,
                        "top1": f"{top1_acc:.3f}",
                        "top3": f"{top3_acc:.3f}",
                    }
                )
                pbar.update(1)

        # Select best model (considering both validation and CV scores)
        # Select best model based on top1 accuracy only
        best_model_name = max(results.keys(), key=lambda x: results[x]["top1_accuracy"])
        self.best_model = self.models[best_model_name]

        print("\n📈 Evaluation Results:")
        for name in results:
            r = results[name]
            print(f"  {name}:")
            # Handle CV score formatting (could be string or float)
            cv_score_str = (
                f"{r['cv_score']:.4f}"
                if isinstance(r["cv_score"], (int, float))
                else str(r["cv_score"])
            )
            print(
                f"    Top-1: {r['top1_accuracy']:.4f} | Top-3: {r['top3_accuracy']:.4f} | CV: {cv_score_str}"
            )

        print(f"\n🏆 Best model: {best_model_name}")
        best_results = results[best_model_name]
        print(f"   Top-1 Accuracy: {best_results['top1_accuracy']:.4f}")
        print(f"   Top-3 Accuracy: {best_results['top3_accuracy']:.4f}")

        return results, best_model_name

    def generate_reports(self, X_val, y_val, results, best_model_name):
        """Generate comprehensive evaluation reports"""
        print("\nGenerating evaluation reports...")

        # Create output directory
        os.makedirs(f"{self.data_dir}/reports", exist_ok=True)

        # Classification report for best model
        best_pred = results[best_model_name]["predictions"]
        class_names = self.label_encoder.classes_

        print(f"\nClassification Report for {best_model_name}:")
        report = classification_report(y_val, best_pred, target_names=class_names)
        print(report)

        # Save classification report
        with open(f"{self.data_dir}/reports/classification_report.txt", "w") as f:
            f.write(f"Classification Report for {best_model_name}\n")
            f.write("=" * 50 + "\n")
            f.write(report)

        # Confusion Matrix
        cm = confusion_matrix(y_val, best_pred)

        plt.figure(figsize=(16, 12))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
        )
        plt.title(f"Confusion Matrix - {best_model_name}\n(Augmented Data)")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(
            f"{self.data_dir}/reports/confusion_matrix_{best_model_name.lower()}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # Model comparison plots
        model_names = list(results.keys())
        top1_scores = [results[name]["top1_accuracy"] for name in model_names]
        top3_scores = [results[name]["top3_accuracy"] for name in model_names]
        # Convert CV scores to float, use 0.0 for non-numeric values
        cv_scores = []
        for name in model_names:
            cv_score = results[name]["cv_score"]
            if isinstance(cv_score, (int, float)):
                cv_scores.append(cv_score)
            else:
                cv_scores.append(0.0)  # Default for "N/A (direct training)"

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

        # Top-1 accuracy comparison
        bars1 = ax1.bar(
            model_names, top1_scores, color=["#ff7f0e", "#2ca02c", "#1f77b4", "#d62728"]
        )
        ax1.set_title("Top-1 Accuracy Comparison\n(Augmented Data)")
        ax1.set_ylabel("Accuracy")
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis="x", rotation=45)
        for bar, score in zip(bars1, top1_scores):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                
                f"{score:.3f}",
                ha="center",
                va="bottom",
            )

        # Top-3 accuracy comparison
        bars2 = ax2.bar(
            model_names, top3_scores, color=["#ff7f0e", "#2ca02c", "#1f77b4", "#d62728"]
        )
        ax2.set_title("Top-3 Accuracy Comparison\n(Augmented Data)")
        ax2.set_ylabel("Accuracy")
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis="x", rotation=45)
        for bar, score in zip(bars2, top3_scores):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{score:.3f}",
                ha="center",
                va="bottom",
            )

        # CV score comparison
        bars3 = ax3.bar(
            model_names, cv_scores, color=["#ff7f0e", "#2ca02c", "#1f77b4", "#d62728"]
        )
        ax3.set_title("Cross-Validation Score Comparison\n(Augmented Data)")
        ax3.set_ylabel("CV Score")
        ax3.set_ylim(0, 1)
        ax3.tick_params(axis="x", rotation=45)
        for bar, score in zip(bars3, cv_scores):
            ax3.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{score:.3f}",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()
        plt.savefig(
            f"{self.data_dir}/reports/model_comparison.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # Save results summary
        results_summary = {
            "augmented_data": self.use_augmented_data,
            "best_model": best_model_name,
            "model_performance": {
                name: {
                    "top1_accuracy": float(results[name]["top1_accuracy"]),
                    "top3_accuracy": float(results[name]["top3_accuracy"]),
                    "cv_score": float(
                        results[name]["cv_score"]
                        if isinstance(results[name]["cv_score"], (int, float))
                        else 0.0
                    ),
                }
                for name in results.keys()
            },
            "feature_selection": {
                "n_features_selected": int(self.feature_selector.k),
                "total_features_before_selection": X_val.shape[1]
                + self.feature_selector.k,
            },
        }

        with open(f"{self.data_dir}/reports/results_summary.json", "w") as f:
            json.dump(results_summary, f, indent=2)

        print(f"📊 Reports saved to {self.data_dir}/reports/")

    def save_model(self):
        """Save trained models and preprocessors"""
        print("Saving trained models...")

        # Save best model
        with open(f"{self.data_dir}/best_model.pkl", "wb") as f:
            pickle.dump(self.best_model, f)

        # Save all preprocessors
        with open(f"{self.data_dir}/scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)

        with open(f"{self.data_dir}/variance_selector.pkl", "wb") as f:
            pickle.dump(self.variance_selector, f)

        with open(f"{self.data_dir}/feature_selector.pkl", "wb") as f:
            pickle.dump(self.feature_selector, f)

        with open(f"{self.data_dir}/label_encoder.pkl", "wb") as f:
            pickle.dump(self.label_encoder, f)

        print("✅ Models and preprocessors saved successfully!")

    def predict_test_set(self):
        """Generate predictions for test set"""
        print("Generating predictions for test set...")

        # Import here to avoid circular imports
        if self.use_augmented_data:
            from task1_preprocessing_augmented import AugmentedAudioFeatureExtractor

            extractor = AugmentedAudioFeatureExtractor(use_augmentation=False)
        else:
            from task1_preprocessing import extract_features_parallel

        test_files = [f"{i:03d}.mp3" for i in range(1, 234)]
        test_paths = [f"test/{file}" for file in test_files]

        print("Extracting test features...")
        if self.use_augmented_data:
            # Create dummy augmented data for test (just original)
            test_data = []
            import torchaudio
            import torchaudio.transforms as T

            for path in test_paths:
                full_path = f"data/artist20/{path}"
                if os.path.exists(full_path):
                    try:
                        waveform, sr = torchaudio.load(full_path)
                        if sr != 16000:
                            resampler = T.Resample(sr, 16000)
                            waveform = resampler(waveform)
                        if waveform.shape[0] > 1:
                            waveform = torch.mean(waveform, dim=0, keepdim=True)
                        test_data.append(
                            {
                                "waveform": waveform,
                                "label": "unknown",
                                "augmentation_type": "original",
                            }
                        )
                    except:
                        continue

            X_test, _ = extractor.extract_features_from_augmented_data(test_data)
        else:
            X_test, _ = extract_features_parallel(test_paths, base_dir="data/artist20/")

        # Preprocess test features the same way as training data
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e6, neginf=-1e6)
        X_test = self.variance_selector.transform(X_test)
        X_test_scaled = self.scaler.transform(X_test)
        X_test_selected = self.feature_selector.transform(X_test_scaled)

        # Generate predictions
        y_pred_proba = self.best_model.predict_proba(X_test_selected)

        # Get top-3 predictions
        top3_indices = np.argsort(y_pred_proba, axis=1)[:, -3:][:, ::-1]
        top3_labels = self.label_encoder.inverse_transform(
            top3_indices.flatten()
        ).reshape(-1, 3)

        # Format predictions
        predictions = {}
        for i, (file, preds) in enumerate(zip(test_files, top3_labels)):
            file_id = file.split(".")[0]
            predictions[file_id] = preds.tolist()

        # Save predictions
        output_file = f"{self.data_dir}/test_predictions.json"
        with open(output_file, "w") as f:
            json.dump(predictions, f, indent=2)

        print(f"🎯 Test predictions saved to {output_file}")
        return predictions


def main():
    """Main training pipeline"""
    print("🎵 Task 1: Augmented Traditional ML Pipeline")
    print("=" * 50)

    # Check for augmented data
    use_augmented = os.path.exists("results/task1_augmented/X_train.npy")

    if not use_augmented:
        print("Augmented data not found. Using original data...")
        print("Run task1_preprocessing_augmented.py to create augmented dataset.")

    # Initialize pipeline
    pipeline = AugmentedTraditionalMLPipeline(use_augmented_data=use_augmented)

    try:
        # Load data
        X_train, y_train, X_val, y_val, label_mapping = pipeline.load_data()

        # Preprocess data
        (
            X_train_processed,
            y_train_encoded,
            X_val_processed,
            y_val_encoded,
            class_weights,
        ) = pipeline.preprocess_data(X_train, y_train, X_val, y_val)

        # Train models
        pipeline.train_models(X_train_processed, y_train_encoded, class_weights)

        # Evaluate models
        results, best_model_name = pipeline.evaluate_models(
            X_val_processed, y_val_encoded
        )

        # Generate reports
        pipeline.generate_reports(
            X_val_processed, y_val_encoded, results, best_model_name
        )

        # Save model
        pipeline.save_model()

        # Generate test predictions
        test_predictions = pipeline.predict_test_set()

        print("\n✅ Task 1 Augmented Implementation Completed Successfully!")
        print("=" * 50)
        print(f"📁 Results saved to: {pipeline.data_dir}/")
        print("📊 Key outputs:")
        print("  - Model comparison plots")
        print("  - Confusion matrix")
        print("  - Classification report")
        print("  - Test set predictions")
        print(f"  - Best model: {best_model_name}")

    except Exception as e:
        print(f"❌ Error in pipeline: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
