"""Base interface for music generation models."""

from abc import ABC, abstractmethod
import numpy as np


class BaseGenerator(ABC):
    """Abstract base class for music generation models."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        duration: float = 30.0,
        **kwargs
    ) -> np.ndarray:
        """
        Generate music from text prompt.

        Args:
            prompt: Text description for music generation
            duration: Duration of generated music in seconds
            **kwargs: Additional generation parameters

        Returns:
            Generated audio as numpy array
        """
        pass

    @abstractmethod
    def generate_with_melody(
        self,
        prompt: str,
        melody: np.ndarray,
        duration: float = 30.0,
        **kwargs
    ) -> np.ndarray:
        """
        Generate music from text and melody conditioning.

        Args:
            prompt: Text description for music generation
            melody: Melody audio as numpy array for conditioning
            duration: Duration of generated music in seconds
            **kwargs: Additional generation parameters

        Returns:
            Generated audio as numpy array
        """
        pass

    @abstractmethod
    def save_audio(self, audio: np.ndarray, output_path: str, sr: int = 24000):
        """
        Save generated audio to file.

        Args:
            audio: Audio array to save
            output_path: Output file path
            sr: Sample rate
        """
        pass
