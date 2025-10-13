"""Meta Audiobox Aesthetics metric for audio quality evaluation."""

import torch
import torchaudio
from pathlib import Path

try:
    from audiobox import AudioboxMetrics
    AUDIOBOX_AVAILABLE = True
except ImportError:
    AUDIOBOX_AVAILABLE = False
    print("Warning: audiobox not available. Install with: pip install audiobox")


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
        if not AUDIOBOX_AVAILABLE:
            raise ImportError(
                "audiobox package not installed. "
                "Please install with: pip install audiobox"
            )

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading Audiobox Aesthetics model on {self.device}...")

        # Initialize Audiobox metrics model
        self.model = AudioboxMetrics(device=self.device)

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
        # Load audio
        audio, sr = torchaudio.load(audio_path)

        # Get aesthetics scores
        results = self.model.evaluate(audio, sr)

        return {
            "ce": float(results.get("content_enjoyment", 0)),
            "cu": float(results.get("content_usefulness", 0)),
            "pc": float(results.get("production_complexity", 0)),
            "pq": float(results.get("production_quality", 0)),
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


class DummyAestheticsMetric:
    """
    Dummy implementation of AestheticsMetric for testing without audiobox.

    Returns random scores for demonstration purposes.
    """

    def __init__(self, device=None):
        print("Using dummy Aesthetics metric (audiobox not available)")
        import random
        self.random = random

    def evaluate_audio(self, audio_path: str) -> dict:
        """Return dummy aesthetics scores."""
        import random

        return {
            "ce": round(random.uniform(0.5, 0.9), 3),
            "cu": round(random.uniform(0.5, 0.9), 3),
            "pc": round(random.uniform(0.5, 0.9), 3),
            "pq": round(random.uniform(0.5, 0.9), 3),
            "audio_file": Path(audio_path).name,
        }

    def evaluate_batch(self, audio_paths: list) -> list:
        """Evaluate multiple audio files with dummy scores."""
        results = []
        for audio_path in audio_paths:
            print(f"Evaluating aesthetics (dummy): {Path(audio_path).name}")
            result = self.evaluate_audio(audio_path)
            results.append(result)

        return results


def get_aesthetics_metric(device=None, use_dummy=False):
    """
    Factory function to get aesthetics metric.

    Args:
        device: torch device
        use_dummy: If True, use dummy implementation

    Returns:
        AestheticsMetric or DummyAestheticsMetric instance
    """
    if use_dummy or not AUDIOBOX_AVAILABLE:
        return DummyAestheticsMetric(device)
    return AestheticsMetric(device)
