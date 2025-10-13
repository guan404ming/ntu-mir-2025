"""CLAP (Contrastive Language-Audio Pretraining) encoder for audio embedding."""

import torch
import sys
from .base_encoder import BaseAudioEncoder


class CLAPEncoder(BaseAudioEncoder):
    """CLAP encoder for extracting audio embeddings."""

    def __init__(self, model_name=None, device=None):
        """
        Initialize CLAP encoder.

        Args:
            model_name: CLAP model checkpoint path (optional, will download default if None)
            device: torch device (cuda/cpu), auto-detect if None
        """
        # Save and clear sys.argv to avoid conflicts with laion_clap's argparse
        _argv = sys.argv.copy()
        sys.argv = [sys.argv[0]]

        # Monkey-patch torch.load to use weights_only=False for CLAP compatibility
        _original_load = torch.load

        def _patched_load(*args, **kwargs):
            if "weights_only" not in kwargs:
                kwargs["weights_only"] = False
            return _original_load(*args, **kwargs)

        torch.load = _patched_load

        try:
            import laion_clap

            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Loading CLAP model on {self.device}...")

            # Initialize CLAP model
            self.model = laion_clap.CLAP_Module(enable_fusion=False, device=self.device)

            # Patch model.load_state_dict to use strict=False
            _original_load_state = self.model.model.load_state_dict

            def _patched_load_state(state_dict, strict=True):
                return _original_load_state(state_dict, strict=False)

            self.model.model.load_state_dict = _patched_load_state

            # Load checkpoint
            if model_name:
                print(f"Loading checkpoint: {model_name}")
                self.model.load_ckpt(model_name)
            else:
                # Use default checkpoint from HuggingFace
                print("Loading default CLAP model from HuggingFace...")
                self.model.load_ckpt()

            # Restore original method
            self.model.model.load_state_dict = _original_load_state

            print("CLAP model loaded successfully")

        finally:
            # Restore torch.load and sys.argv
            torch.load = _original_load
            sys.argv = _argv

    def encode_audio(self, audio_path):
        """
        Encode audio file to embedding vector.

        Args:
            audio_path: Path to audio file

        Returns:
            numpy array of shape (embed_dim,) containing audio embedding
        """
        # Get audio embedding
        audio_embed = self.model.get_audio_embedding_from_filelist(x=[audio_path])

        # Convert to numpy if tensor
        if hasattr(audio_embed, "cpu"):
            audio_embed = audio_embed.cpu().numpy()

        # Returns shape (1, embed_dim), squeeze to (embed_dim,)
        return audio_embed[0]

    def encode_text(self, text):
        """
        Encode text to embedding vector.

        Args:
            text: Text string or list of text strings

        Returns:
            numpy array containing text embedding(s)
        """
        if isinstance(text, str):
            text = [text]

        text_embed = self.model.get_text_embedding(text)

        # Convert to numpy if tensor
        if hasattr(text_embed, "cpu"):
            text_embed = text_embed.cpu().numpy()

        if len(text) == 1:
            return text_embed[0]
        return text_embed
