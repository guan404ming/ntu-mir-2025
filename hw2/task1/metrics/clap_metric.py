"""CLAP similarity metric for evaluation."""

from pathlib import Path


class CLAPMetric:
    """Calculate CLAP cosine similarity between audio pairs."""

    def __init__(self, clap_encoder):
        """
        Initialize CLAP metric.

        Args:
            clap_encoder: Initialized CLAP encoder instance
        """
        self.encoder = clap_encoder

    def calculate_similarity(self, audio1_path: str, audio2_path: str) -> float:
        """
        Calculate CLAP cosine similarity between two audio files.

        Args:
            audio1_path: Path to first audio file
            audio2_path: Path to second audio file

        Returns:
            Cosine similarity score (float)
        """
        # Encode both audio files
        embed1 = self.encoder.encode_audio(audio1_path)
        embed2 = self.encoder.encode_audio(audio2_path)

        # Calculate cosine similarity
        similarity = self.encoder.cosine_similarity(embed1, embed2)

        return similarity

    def evaluate_retrieval(
        self, target_path: str, retrieved_path: str
    ) -> dict:
        """
        Evaluate retrieval quality using CLAP similarity.

        Args:
            target_path: Path to target audio
            retrieved_path: Path to retrieved/generated audio

        Returns:
            Dictionary with evaluation results
        """
        similarity = self.calculate_similarity(target_path, retrieved_path)

        return {
            "clap_similarity": similarity,
            "target": Path(target_path).name,
            "retrieved": Path(retrieved_path).name,
        }
