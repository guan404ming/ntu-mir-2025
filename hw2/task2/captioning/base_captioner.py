"""Base interface for audio captioning models."""

from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseCaptioner(ABC):
    """Abstract base class for audio captioning models."""

    @abstractmethod
    def generate_caption(self, audio_path: str, **kwargs) -> str:
        """
        Generate a text caption describing the audio.

        Args:
            audio_path: Path to the audio file
            **kwargs: Additional generation parameters

        Returns:
            Generated text caption
        """
        pass

    @abstractmethod
    def generate_detailed_caption(self, audio_path: str) -> Dict[str, Any]:
        """
        Generate detailed caption with additional metadata.

        Args:
            audio_path: Path to the audio file

        Returns:
            Dictionary containing caption and metadata
        """
        pass
