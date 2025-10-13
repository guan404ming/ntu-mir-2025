"""Music2latent encoder for audio embedding."""

import torch
import numpy as np
import librosa


class Music2LatentEncoder:
    """Music2latent encoder for extracting audio embeddings."""

    def __init__(self, device=None):
        """
        Initialize Music2latent encoder.

        Args:
            device: torch device (cuda/cpu), auto-detect if None
        """
        try:
            from music2latent import EncoderDecoder
        except ImportError:
            raise ImportError(
                "music2latent is not installed. Install with: pip install music2latent"
            )

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading Music2latent model on {self.device}...")

        # Monkey-patch torch.load to use weights_only=False for compatibility
        _original_load = torch.load
        def _patched_load(*args, **kwargs):
            if 'weights_only' not in kwargs:
                kwargs['weights_only'] = False
            return _original_load(*args, **kwargs)
        torch.load = _patched_load

        try:
            # Initialize encoder-decoder
            self.model = EncoderDecoder(device=self.device)
            print("Music2latent model loaded successfully")
        finally:
            # Restore original torch.load
            torch.load = _original_load

    def encode_audio(self, audio_path):
        """
        Encode audio file to latent embedding.

        Args:
            audio_path: Path to audio file

        Returns:
            numpy array containing latent audio embedding
        """
        # Load audio at 44.1kHz (music2latent's expected sample rate)
        wv, sr = librosa.load(audio_path, sr=44100, mono=True)

        # Encode to latent representation
        # Returns shape: (1, 64, sequence_length)
        latent = self.model.encode(wv)

        # Convert to numpy if tensor
        if isinstance(latent, torch.Tensor):
            latent = latent.cpu().numpy()

        # Pool over time dimension to get fixed-size embedding
        # Shape: (1, 64, seq_len) -> (64,)
        # Using mean pooling to get a single embedding vector
        embedding = latent.mean(axis=-1).squeeze()

        return embedding

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
        # Normalize embeddings
        embed1_norm = embed1 / (np.linalg.norm(embed1) + 1e-8)
        embed2_norm = embed2 / (np.linalg.norm(embed2) + 1e-8)

        # Compute cosine similarity
        return float(np.dot(embed1_norm, embed2_norm))
