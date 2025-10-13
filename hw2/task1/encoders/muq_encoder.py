"""MuQ (Music Quantization) encoder for audio embedding."""

import torch
import librosa
from .base_encoder import BaseAudioEncoder


class MuQEncoder(BaseAudioEncoder):
    """MuQ encoder for extracting audio embeddings."""

    def __init__(self, model_name="OpenMuQ/MuQ-large-msd-iter", device=None):
        """
        Initialize MuQ encoder.

        Args:
            model_name: MuQ model identifier from HuggingFace
            device: torch device (cuda/cpu), auto-detect if None
        """
        try:
            from muq import MuQ
        except ImportError:
            raise ImportError("muq is not installed. Install with: pip install muq")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading MuQ model on {self.device}...")

        # Load pretrained MuQ model
        self.model = MuQ.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()

        # MuQ expects 24kHz audio
        self.sample_rate = 24000

        print("MuQ model loaded successfully")

    def encode_audio(self, audio_path):
        """
        Encode audio file to embedding using MuQ.

        Args:
            audio_path: Path to audio file

        Returns:
            numpy array containing audio embedding
        """
        # Load audio at 24kHz (MuQ's expected sample rate)
        wv, _ = librosa.load(audio_path, sr=self.sample_rate, mono=True)

        # Convert to tensor and add batch dimension
        wavs = torch.tensor(wv).unsqueeze(0).to(self.device)

        # Extract features using MuQ
        with torch.no_grad():
            output = self.model(wavs, output_hidden_states=True)

            # Get the last hidden state
            # Shape: (batch_size, sequence_length, hidden_dim)
            hidden_state = output.last_hidden_state

            # Pool over time dimension to get fixed-size embedding
            # Using mean pooling across the sequence
            embedding = hidden_state.mean(dim=1)  # (batch_size, hidden_dim)

            # Convert to numpy
            embedding = embedding.cpu().numpy()[0]

        return embedding
