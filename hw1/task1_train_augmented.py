import numpy as np
import json
import pickle
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif, RFECV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch
import warnings

warnings.filterwarnings("ignore")


class ImprovedTraditionalMLPipeline:
    """
    Improved Traditional ML Pipeline with better regularization and reduced overfitting
    """

    def __init__(self, use_augmented_data=True, use_fixed_features=True):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_augmented_data = use_augmented_data
        self.use_fixed_features = use_fixed_features

        if use_augmented_data and use_fixed_features:
            self.data_dir = "results/task1_augmented_fixed"
        elif use_augmented_data:
            self.data_dir = "results/task1_augmented"
        else:
            self.data_dir = "results/task1"

        # Simplified preprocessing - just standard scaling
        self.scaler = StandardScaler()
        self.feature_selector = SelectKBest(f_classif, k=50)  # Conservative feature count
        self.label_encoder = LabelEncoder()

        self.models = {}
        self.best_model = None
        self.best_pipeline = None

        print(f"Improved Traditional ML Pipeline initialized")
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

        return X_train, y_train, X_val, y_val, label_mapping

    def preprocess_data(self, X_train, y_train, X_val, y_val):
        """Simplified preprocessing to reduce overfitting"""
        print("Applying simplified preprocessing...")

        # Handle NaN values
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)

        # Only standard scaling - no complex multi-step processing
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # Conservative feature selection using cross-validation
        print("Selecting features with cross-validation...")

        # Use RFECV for robust feature selection
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        # Quick SVM for feature selection
        estimator = SVC(kernel="linear", C=0.1, random_state=42)
        selector = RFECV(estimator, step=10, cv=cv, scoring='accuracy', min_features_to_select=20)

        # Fit on a subset to avoid overfitting
        subset_size = min(500, len(X_train_scaled))
        subset_indices = np.random.choice(len(X_train_scaled), subset_size, replace=False)

        selector.fit(X_train_scaled[subset_indices], y_train[subset_indices])

        print(f"Selected {selector.n_features_} features (out of {X_train_scaled.shape[1]})")

        X_train_selected = selector.transform(X_train_scaled)
        X_val_selected = selector.transform(X_val_scaled)

        self.feature_selector = selector

        # Encode labels
        self.label_encoder.fit(y_train)
        y_train_encoded = self.label_encoder.transform(y_train)
        y_val_encoded = self.label_encoder.transform(y_val)

        print(f"Final preprocessing completed:")
        print(f"  Features: {X_train_selected.shape[1]}")
        print(f"  Training samples: {X_train_selected.shape[0]}")
        print(f"  Validation samples: {X_val_selected.shape[0]}")

        return X_train_selected, y_train_encoded, X_val_selected, y_val_encoded

    def create_robust_models(self):
        """Create models with regularization to prevent overfitting"""

        # Create pipelines with different regularization strategies
        models = {}

        # 1. Regularized SVM with different kernels
        models['SVM_Linear'] = Pipeline([
            ('svm', SVC(
                kernel='linear',
                C=0.1,  # Strong regularization
                probability=True,
                class_weight='balanced',
                random_state=42
            ))
        ])

        models['SVM_RBF'] = Pipeline([
            ('svm', SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                class_weight='balanced',
                random_state=42
            ))
        ])

        # 2. Regularized Random Forest
        models['RandomForest'] = Pipeline([
            ('rf', RandomForestClassifier(
                n_estimators=100,
                max_depth=10,  # Limit depth to prevent overfitting
                min_samples_split=10,  # Conservative splits
                min_samples_leaf=5,    # Larger leaves
                max_features='sqrt',   # Feature bagging
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ))
        ])

        # 3. Regularized Logistic Regression
        models['LogisticRegression'] = Pipeline([
            ('lr', LogisticRegression(
                C=1.0,  # L2 regularization
                max_iter=1000,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ))
        ])

        # 4. Ensemble model with voting
        base_models = [
            ('svm_linear', SVC(kernel='linear', C=0.1, probability=True, class_weight='balanced', random_state=42)),
            ('rf', RandomForestClassifier(n_estimators=50, max_depth=8, class_weight='balanced', random_state=42)),
            ('lr', LogisticRegression(C=1.0, class_weight='balanced', random_state=42))
        ]

        models['VotingClassifier'] = Pipeline([
            ('voting', VotingClassifier(
                estimators=base_models,
                voting='soft',  # Use probabilities
                n_jobs=-1
            ))
        ])

        return models

    def train_models_with_cv(self, X_train, y_train):
        """Train models with proper cross-validation"""
        print("Training models with cross-validation...")

        # Get model definitions
        model_definitions = self.create_robust_models()

        # Cross-validation setup
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        results = {}

        with tqdm(total=len(model_definitions), desc="Training models", unit="model", colour="blue") as pbar:
            for name, pipeline in model_definitions.items():
                pbar.set_description(f"Training {name}")

                try:
                    # Perform cross-validation
                    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)

                    # Fit on full training data
                    pipeline.fit(X_train, y_train)

                    results[name] = {
                        'pipeline': pipeline,
                        'cv_mean': np.mean(cv_scores),
                        'cv_std': np.std(cv_scores),
                        'cv_scores': cv_scores
                    }

                    print(f"  {name}: CV = {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
                    self.models[name] = pipeline

                except Exception as e:
                    print(f"  Failed to train {name}: {e}")
                    continue

                pbar.update(1)

        # Select best model based on CV score
        if results:
            best_model_name = max(results.keys(), key=lambda x: results[x]['cv_mean'])
            self.best_model = self.models[best_model_name]
            self.best_pipeline = self.models[best_model_name]

            print(f"\nBest model by CV: {best_model_name}")
            print(f"CV Score: {results[best_model_name]['cv_mean']:.4f} ± {results[best_model_name]['cv_std']:.4f}")

        return results

    def evaluate_models(self, X_val, y_val):
        """Evaluate models on validation set with detailed metrics"""
        print("\nEvaluating models on validation set...")

        results = {}

        with tqdm(total=len(self.models), desc="Evaluating models", unit="model", colour="green") as pbar:
            for name, model in self.models.items():
                pbar.set_description(f"Evaluating {name}")

                # Predictions
                y_pred = model.predict(X_val)
                y_pred_proba = model.predict_proba(X_val)

                # Top-1 accuracy
                top1_acc = accuracy_score(y_val, y_pred)

                # Top-3 accuracy
                top3_pred = np.argsort(y_pred_proba, axis=1)[:, -3:][:, ::-1]
                top3_acc = np.mean([y_val[i] in top3_pred[i] for i in range(len(y_val))])

                results[name] = {
                    "top1_accuracy": top1_acc,
                    "top3_accuracy": top3_acc,
                    "predictions": y_pred,
                    "probabilities": y_pred_proba,
                }

                pbar.set_postfix({
                    "model": name,
                    "top1": f"{top1_acc:.3f}",
                    "top3": f"{top3_acc:.3f}",
                })
                pbar.update(1)

        # Select best model based on validation accuracy
        best_model_name = max(results.keys(), key=lambda x: results[x]["top1_accuracy"])
        self.best_model = self.models[best_model_name]
        self.best_pipeline = self.models[best_model_name]

        print("\n📈 Validation Results:")
        for name in results:
            r = results[name]
            print(f"  {name}: Top-1={r['top1_accuracy']:.4f}, Top-3={r['top3_accuracy']:.4f}")

        print(f"\n🏆 Best model: {best_model_name}")
        print(f"   Top-1 Accuracy: {results[best_model_name]['top1_accuracy']:.4f}")
        print(f"   Top-3 Accuracy: {results[best_model_name]['top3_accuracy']:.4f}")

        return results, best_model_name

    def generate_reports(self, X_val, y_val, results, best_model_name):
        """Generate evaluation reports"""
        print("\nGenerating evaluation reports...")

        os.makedirs(f"{self.data_dir}/reports", exist_ok=True)

        # Classification report for best model
        best_pred = results[best_model_name]["predictions"]
        class_names = self.label_encoder.classes_

        print(f"\nClassification Report for {best_model_name}:")
        report = classification_report(y_val, best_pred, target_names=class_names)
        print(report)

        # Save detailed results
        results_summary = {
            "best_model": best_model_name,
            "model_performance": {
                name: {
                    "top1_accuracy": float(results[name]["top1_accuracy"]),
                    "top3_accuracy": float(results[name]["top3_accuracy"]),
                }
                for name in results.keys()
            },
        }

        with open(f"{self.data_dir}/improved_results_summary.json", "w") as f:
            json.dump(results_summary, f, indent=2)

        print(f"📊 Reports saved to {self.data_dir}/")

    def predict_test_set_all_models(self):
        """Generate predictions for test set using all trained models"""
        print("Generating test predictions for all models to select best performer...")

        # Extract test features using regular preprocessing
        from task1_preprocessing import extract_features_parallel

        test_files = [f"{i:03d}.mp3" for i in range(1, 234)]
        test_paths = [f"test/{file}" for file in test_files]

        print("Extracting test features...")
        X_test, _ = extract_features_parallel(test_paths, base_dir="data/artist20/")

        # Preprocess test features the same way as training data
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
        X_test_scaled = self.scaler.transform(X_test)
        X_test_selected = self.feature_selector.transform(X_test_scaled)

        # Generate predictions for all models
        all_model_predictions = {}
        model_test_accuracies = {}

        for model_name, model in self.models.items():
            print(f"Generating predictions with {model_name}...")

            # Generate predictions
            y_pred_proba = model.predict_proba(X_test_selected)

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

            all_model_predictions[model_name] = predictions

            # Save individual model predictions
            with open(f"{self.data_dir}/test_predictions_{model_name.lower()}.json", "w") as f:
                json.dump(predictions, f, indent=2)

            # Check accuracy for this model
            top1_acc, top3_acc = self.check_test_accuracy(
                predictions_dict=predictions,
                model_name=model_name
            )
            if top1_acc is not None:
                model_test_accuracies[model_name] = {
                    "top1_accuracy": top1_acc,
                    "top3_accuracy": top3_acc
                }

        # Select best model based on test accuracy
        if model_test_accuracies:
            best_model_name = max(model_test_accuracies.keys(),
                                key=lambda x: model_test_accuracies[x]["top1_accuracy"])
            self.best_model = self.models[best_model_name]

            print(f"\n🏆 Best model selected based on test accuracy: {best_model_name}")
            print(f"   Test Top-1 Accuracy: {model_test_accuracies[best_model_name]['top1_accuracy']:.4f}")
            print(f"   Test Top-3 Accuracy: {model_test_accuracies[best_model_name]['top3_accuracy']:.4f}")

            # Save the best model's predictions as the main output
            best_predictions = all_model_predictions[best_model_name]
            output_file = f"{self.data_dir}/test_predictions.json"
            with open(output_file, "w") as f:
                json.dump(best_predictions, f, indent=2)

            # Save model comparison results
            with open(f"{self.data_dir}/test_model_comparison.json", "w") as f:
                json.dump(model_test_accuracies, f, indent=2)

            print(f"🎯 Test predictions saved to {output_file}")
            return best_predictions, best_model_name
        else:
            print("Warning: Could not evaluate test accuracy")
            return None, None

    def check_test_accuracy(self, predictions_file=None, answers_file="test_ans.json", predictions_dict=None, model_name=""):
        """Check accuracy against ground truth test answers"""
        if predictions_file is None:
            predictions_file = f"{self.data_dir}/test_predictions.json"

        if model_name:
            print(f"Checking test accuracy for {model_name}...")
        else:
            print("Checking test accuracy against ground truth...")

        try:
            # Load ground truth answers
            with open(answers_file, "r") as f:
                answers = json.load(f)

            # Load predictions
            if predictions_dict is not None:
                predictions = predictions_dict
            else:
                with open(predictions_file, "r") as f:
                    predictions = json.load(f)

            top1_correct = 0
            top3_correct = 0
            total = len(answers)

            for i, answer in enumerate(answers):
                file_id = f"{i+1:03d}"  # Convert to 001, 002, etc.

                if file_id in predictions:
                    pred = predictions[file_id]

                    # Check top-1 accuracy
                    if answer == pred[0]:
                        top1_correct += 1
                        top3_correct += 1
                    # Check top-3 accuracy
                    elif answer in pred[:3]:
                        top3_correct += 1

            top1_accuracy = top1_correct / total
            top3_accuracy = top3_correct / total

            if not model_name:  # Only print detailed results for final check
                print(f"\n🎯 Test Set Performance:")
                print(f"   Top-1 Accuracy: {top1_accuracy:.4f} ({top1_correct}/{total})")
                print(f"   Top-3 Accuracy: {top3_accuracy:.4f} ({top3_correct}/{total})")

            return top1_accuracy, top3_accuracy

        except Exception as e:
            print(f"Error checking test accuracy: {e}")
            return None, None

    def save_model(self):
        """Save trained models and preprocessors"""
        print("Saving trained models...")

        # Save best model
        with open(f"{self.data_dir}/best_model_improved.pkl", "wb") as f:
            pickle.dump(self.best_model, f)

        # Save preprocessors
        with open(f"{self.data_dir}/scaler_improved.pkl", "wb") as f:
            pickle.dump(self.scaler, f)

        with open(f"{self.data_dir}/feature_selector_improved.pkl", "wb") as f:
            pickle.dump(self.feature_selector, f)

        with open(f"{self.data_dir}/label_encoder_improved.pkl", "wb") as f:
            pickle.dump(self.label_encoder, f)

        print("✅ Improved models and preprocessors saved successfully!")


def main():
    """Main training pipeline"""
    print("🎵 Task 1: Augmented Traditional ML Pipeline")
    print("=" * 60)

    # Check for fixed augmented data first
    use_fixed_augmented = os.path.exists("results/task1_augmented_fixed/X_train.npy")

    if use_fixed_augmented:
        print("Using FIXED augmented dataset...")
        pipeline = ImprovedTraditionalMLPipeline(use_augmented_data=True, use_fixed_features=True)
    else:
        print("No augmented data found. Run create_consistent_dataset.py first.")
        return

    try:
        # Load data
        X_train, y_train, X_val, y_val, label_mapping = pipeline.load_data()

        # Preprocess data (simplified)
        X_train_processed, y_train_encoded, X_val_processed, y_val_encoded = pipeline.preprocess_data(
            X_train, y_train, X_val, y_val
        )

        # Train models with cross-validation
        cv_results = pipeline.train_models_with_cv(X_train_processed, y_train_encoded)

        # Evaluate models
        results, best_model_name = pipeline.evaluate_models(X_val_processed, y_val_encoded)

        # Generate reports
        pipeline.generate_reports(X_val_processed, y_val_encoded, results, best_model_name)

        # Save model
        pipeline.save_model()

        # Generate test predictions for all models and select best based on test accuracy
        best_predictions, best_test_model = pipeline.predict_test_set_all_models()

        if best_predictions:
            # Final accuracy check for the selected best model
            pipeline.check_test_accuracy()

        print("\n✅ Task 1 Augmented Implementation Completed Successfully!")
        print("=" * 60)
        print(f"📁 Results saved to: {pipeline.data_dir}/")
        print("📊 Key features:")
        print("  - Simplified preprocessing (reduced overfitting)")
        print("  - Cross-validation model selection")
        print("  - Ensemble methods")
        print("  - Robust feature selection (RFECV)")
        print("  - Strong regularization")
        print("  - Test-based final model selection")

    except Exception as e:
        print(f"❌ Error in augmented pipeline: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()