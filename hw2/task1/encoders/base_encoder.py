"""Base encoder interface for audio embedding models."""

from abc import ABC, abstractmethod


class BaseAudioEncoder(ABC):
    """Abstract base class for audio encoders."""

    @abstractmethod
    def encode_audio(self, audio_path):
        """
        Encode audio file to embedding vector.

        Args:
            audio_path: Path to audio file

        Returns:
            numpy array containing audio embedding
        """
        pass

    @staticmethod
    def cosine_similarity(embed1, embed2):
        """
        Calculate cosine similarity between two embeddings.

        Args:
            embed1: First embedding (numpy array)
            embed2: Second embedding (numpy array)

        Returns:
            Cosine similarity score (float)
        """
        import numpy as np

        # Normalize embeddings
        embed1_norm = embed1 / (np.linalg.norm(embed1) + 1e-8)
        embed2_norm = embed2 / (np.linalg.norm(embed2) + 1e-8)

        # Compute cosine similarity
        return float(np.dot(embed1_norm, embed2_norm))
