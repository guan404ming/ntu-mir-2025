import numpy as np
import json
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    VotingClassifier,
    GradientBoostingClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch
import warnings

warnings.filterwarnings("ignore")


class TraditionalMLPipeline:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        self.feature_selector = SelectKBest(f_classif, k=100)  # Feature selection
        self.label_encoder = LabelEncoder()
        self.models = {}
        self.best_model = None
        self.feature_names = None

        print("Traditional ML models use CPU only (scikit-learn)")

    def load_data(self):
        print("Loading preprocessed features...")
        X_train = np.load("results/task1/X_train.npy")
        y_train = np.load("results/task1/y_train.npy")
        X_val = np.load("results/task1/X_val.npy")
        y_val = np.load("results/task1/y_val.npy")

        with open("results/task1/label_mapping.json", "r") as f:
            label_mapping = json.load(f)

        print(f"Training data: {X_train.shape}")
        print(f"Validation data: {X_val.shape}")
        print(f"Number of classes: {len(label_mapping)}")

        return X_train, y_train, X_val, y_val, label_mapping

    def preprocess_data(self, X_train, y_train, X_val, y_val):
        print("Preprocessing data...")

        # Handle potential NaN values
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)

        # Feature scaling with robust scaler
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # Feature selection to reduce dimensionality and noise
        X_train_scaled = self.feature_selector.fit_transform(X_train_scaled, y_train)
        X_val_scaled = self.feature_selector.transform(X_val_scaled)

        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_val_encoded = self.label_encoder.transform(y_val)

        print(
            f"Feature scaling completed. Mean: {X_train_scaled.mean():.3f}, Std: {X_train_scaled.std():.3f}"
        )

        return X_train_scaled, y_train_encoded, X_val_scaled, y_val_encoded

    def train_models(self, X_train, y_train):
        print("Training traditional ML models...")

        models_to_train = [
            ("SVM", "Support Vector Machine with hyperparameter tuning"),
            ("RandomForest", "Random Forest"),
            ("KNN", "k-Nearest Neighbors"),
        ]

        with tqdm(
            total=len(models_to_train),
            desc="Training models",
            unit="model",
            colour="blue",
        ) as pbar:
            # 1. Support Vector Machine with RBF kernel (best for audio classification)
            pbar.set_description("Training SVM")
            svm_params = {
                "C": [0.1, 1, 10, 100],
                "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
                "kernel": ["rbf"],
            }

            svm = SVC(probability=True, random_state=42)
            svm_grid = GridSearchCV(
                svm, svm_params, cv=5, scoring="accuracy", n_jobs=-1, verbose=0
            )
            svm_grid.fit(X_train, y_train)
            self.models["SVM"] = svm_grid
            pbar.set_postfix({"model": "SVM", "score": f"{svm_grid.best_score_:.4f}"})
            pbar.update(1)

            # 2. Random Forest
            pbar.set_description("Training Random Forest")
            rf_params = {
                "n_estimators": [100, 200, 300],
                "max_depth": [10, 20, 30, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            }

            rf = RandomForestClassifier(random_state=42, n_jobs=-1)
            rf_grid = GridSearchCV(
                rf, rf_params, cv=3, scoring="accuracy", n_jobs=-1, verbose=0
            )
            rf_grid.fit(X_train, y_train)
            self.models["RandomForest"] = rf_grid
            pbar.set_postfix({"model": "RF", "score": f"{rf_grid.best_score_:.4f}"})
            pbar.update(1)

            # 3. k-Nearest Neighbors
            pbar.set_description("Training k-NN")
            knn_params = {
                "n_neighbors": [3, 5, 7, 9, 11, 15],
                "weights": ["uniform", "distance"],
                "metric": ["euclidean", "manhattan", "minkowski"],
            }

            knn = KNeighborsClassifier(n_jobs=-1)
            knn_grid = GridSearchCV(
                knn, knn_params, cv=5, scoring="accuracy", n_jobs=-1, verbose=0
            )
            knn_grid.fit(X_train, y_train)
            self.models["KNN"] = knn_grid
            pbar.set_postfix({"model": "k-NN", "score": f"{knn_grid.best_score_:.4f}"})
            pbar.update(1)

        print(f"\nTraining completed!")
        print(
            f"SVM - Best params: {self.models['SVM'].best_params_}, CV score: {self.models['SVM'].best_score_:.4f}"
        )
        print(
            f"RF - Best params: {self.models['RandomForest'].best_params_}, CV score: {self.models['RandomForest'].best_score_:.4f}"
        )
        print(
            f"k-NN - Best params: {self.models['KNN'].best_params_}, CV score: {self.models['KNN'].best_score_:.4f}"
        )

    def evaluate_models(self, X_val, y_val):
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

                results[name] = {
                    "top1_accuracy": top1_acc,
                    "top3_accuracy": top3_acc,
                    "predictions": y_pred,
                    "probabilities": y_pred_proba,
                }

                pbar.set_postfix(
                    {
                        "model": name,
                        "top1": f"{top1_acc:.3f}",
                        "top3": f"{top3_acc:.3f}",
                    }
                )
                pbar.update(1)

        # Select best model based on top-1 accuracy
        best_model_name = max(results.keys(), key=lambda x: results[x]["top1_accuracy"])
        self.best_model = self.models[best_model_name]

        print(f"\nEvaluation Results:")
        for name in results:
            print(
                f"  {name}: Top-1={results[name]['top1_accuracy']:.4f}, Top-3={results[name]['top3_accuracy']:.4f}"
            )

        print(f"\n🏆 Best model: {best_model_name}")
        print(f"   Top-1 Accuracy: {results[best_model_name]['top1_accuracy']:.4f}")
        print(f"   Top-3 Accuracy: {results[best_model_name]['top3_accuracy']:.4f}")

        return results, best_model_name

    def generate_reports(self, X_val, y_val, results, best_model_name):
        print("\nGenerating evaluation reports...")

        # Classification report for best model
        best_pred = results[best_model_name]["predictions"]
        class_names = self.label_encoder.classes_

        print(f"\nClassification Report for {best_model_name}:")
        print(classification_report(y_val, best_pred, target_names=class_names))

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
        plt.title(f"Confusion Matrix - {best_model_name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(
            f"results/task1/confusion_matrix_{best_model_name.lower()}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # Model comparison
        model_names = list(results.keys())
        top1_scores = [results[name]["top1_accuracy"] for name in model_names]
        top3_scores = [results[name]["top3_accuracy"] for name in model_names]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Top-1 accuracy comparison
        bars1 = ax1.bar(
            model_names, top1_scores, color=["#ff7f0e", "#2ca02c", "#1f77b4"]
        )
        ax1.set_title("Top-1 Accuracy Comparison")
        ax1.set_ylabel("Accuracy")
        ax1.set_ylim(0, 1)
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
            model_names, top3_scores, color=["#ff7f0e", "#2ca02c", "#1f77b4"]
        )
        ax2.set_title("Top-3 Accuracy Comparison")
        ax2.set_ylabel("Accuracy")
        ax2.set_ylim(0, 1)
        for bar, score in zip(bars2, top3_scores):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{score:.3f}",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()
        plt.savefig("results/task1/model_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()

        # Save results
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

        with open("results/task1/results_summary.json", "w") as f:
            json.dump(results_summary, f, indent=2)

        print("Reports saved to results/task1/")

    def save_model(self):
        print("Saving trained models...")

        # Save best model
        with open("results/task1/best_model.pkl", "wb") as f:
            pickle.dump(self.best_model, f)

        # Save preprocessors
        with open("results/task1/scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)

        with open("results/task1/label_encoder.pkl", "wb") as f:
            pickle.dump(self.label_encoder, f)

        print("Models saved successfully!")

    def predict_test_set(self):
        print("Generating predictions for test set...")

        # Extract features for test files
        from task1_preprocessing import extract_features_parallel

        test_files = [f"{i:03d}.mp3" for i in range(1, 234)]
        test_paths = [f"test/{file}" for file in test_files]

        print("Extracting test features...")
        with tqdm(desc="Processing test files", unit="file", colour="blue"):
            X_test, _ = extract_features_parallel(test_paths, base_dir="data/artist20/")

        # Preprocess test features
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
        X_test_scaled = self.scaler.transform(X_test)
        X_test_scaled = self.feature_selector.transform(X_test_scaled)

        # Generate predictions
        y_pred_proba = self.best_model.predict_proba(X_test_scaled)

        # Get top-3 predictions for each test sample
        top3_indices = np.argsort(y_pred_proba, axis=1)[:, -3:][:, ::-1]
        top3_labels = self.label_encoder.inverse_transform(
            top3_indices.flatten()
        ).reshape(-1, 3)

        # Format predictions as required
        predictions = {}
        for i, (file, preds) in enumerate(zip(test_files, top3_labels)):
            file_id = file.split(".")[0]  # Remove .mp3 extension
            predictions[file_id] = preds.tolist()

        # Save predictions
        with open("results/task1/test_predictions.json", "w") as f:
            json.dump(predictions, f, indent=2)

        print(f"Test predictions saved to results/task1/test_predictions.json")
        return predictions


def main():
    # Initialize pipeline
    pipeline = TraditionalMLPipeline()

    try:
        # Load data
        X_train, y_train, X_val, y_val, label_mapping = pipeline.load_data()

        # Preprocess data
        X_train_scaled, y_train_encoded, X_val_scaled, y_val_encoded = (
            pipeline.preprocess_data(X_train, y_train, X_val, y_val)
        )

        # Train models
        pipeline.train_models(X_train_scaled, y_train_encoded)

        # Evaluate models
        results, best_model_name = pipeline.evaluate_models(X_val_scaled, y_val_encoded)

        # Generate reports
        pipeline.generate_reports(X_val_scaled, y_val_encoded, results, best_model_name)

        # Save model
        pipeline.save_model()

        # Generate test predictions
        test_predictions = pipeline.predict_test_set()

        print("\nTask 1 implementation completed successfully!")
        print("Check results/task1/ for all outputs including:")
        print("- Model comparison plots")
        print("- Confusion matrix")
        print("- Performance metrics")
        print("- Test set predictions")

    except Exception as e:
        print(f"Error in pipeline: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
