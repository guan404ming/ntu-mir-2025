"""Meta Audiobox Aesthetics metric for audio quality evaluation."""

import torch
from pathlib import Path


class AestheticsMetric:
    """
    Meta Audiobox Aesthetics metric for evaluating audio quality.

    Measures:
    - CE: Content Enjoyment
    - CU: Content Usefulness
    - PC: Production Complexity
    - PQ: Production Quality
    """

    def __init__(self, device=None):
        """
        Initialize Audiobox Aesthetics metric.

        Args:
            device: torch device (cuda/cpu), auto-detect if None
        """
        try:
            from audiobox_aesthetics.infer import initialize_predictor
        except ImportError:
            raise ImportError(
                "audiobox_aesthetics package not installed. "
                "Please install with: pip install audiobox_aesthetics"
            )

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading Audiobox Aesthetics model on {self.device}...")

        # Initialize Audiobox aesthetics predictor
        self.predictor = initialize_predictor()

        print("Audiobox Aesthetics model loaded successfully")

    def evaluate_audio(self, audio_path: str) -> dict:
        """
        Evaluate audio aesthetics metrics.

        Args:
            audio_path: Path to audio file

        Returns:
            Dictionary with aesthetics scores:
                - ce: Content Enjoyment
                - cu: Content Usefulness
                - pc: Production Complexity
                - pq: Production Quality
        """
        # Predict aesthetics scores
        # The predictor expects a list of dictionaries with "path" key
        results = self.predictor.forward([{"path": audio_path}])

        # Results is a list with one dictionary per input
        # Results format: [{"CE": score, "CU": score, "PC": score, "PQ": score}]
        result = results[0] if isinstance(results, list) else results

        return {
            "ce": float(result.get("CE", 0)),
            "cu": float(result.get("CU", 0)),
            "pc": float(result.get("PC", 0)),
            "pq": float(result.get("PQ", 0)),
            "audio_file": Path(audio_path).name,
        }

    def evaluate_batch(self, audio_paths: list) -> list:
        """
        Evaluate multiple audio files.

        Args:
            audio_paths: List of audio file paths

        Returns:
            List of dictionaries with aesthetics scores
        """
        results = []
        for audio_path in audio_paths:
            print(f"Evaluating aesthetics: {Path(audio_path).name}")
            result = self.evaluate_audio(audio_path)
            results.append(result)

        return results
