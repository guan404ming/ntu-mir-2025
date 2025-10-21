"""MusicGen-based music generator using audiocraft library."""

import torch
import numpy as np
from pathlib import Path
from typing import Optional
from .base_generator import BaseGenerator
from audiocraft.models import MusicGen


class MusicGenGenerator(BaseGenerator):
    """Music generator using Meta's MusicGen model via audiocraft."""

    def __init__(
        self, model_name: str = "facebook/musicgen-medium", device: str = "cuda"
    ):
        """
        Initialize MusicGen generator using audiocraft library.

        Args:
            model_name: Model name (facebook/musicgen-small/medium/large/melody)
            device: Device to use (cuda/cpu)
        """
        self.device = device
        self.model_name = model_name

        print(f"Loading MusicGen model using audiocraft: {model_name}")

        # Extract model size from name (e.g., "facebook/musicgen-medium" -> "medium")
        if "/" in model_name:
            model_size = model_name.split("/")[-1].replace("musicgen-", "")
        else:
            model_size = model_name

        # Check if this is the melody-conditioned model
        self.is_melody_model = "melody" in model_size.lower()

        # Load model using audiocraft
        self.model = MusicGen.get_pretrained(model_size, device=device)

        self.sample_rate = self.model.sample_rate

        print(f"MusicGen model loaded (SR: {self.sample_rate} Hz)")
        print(
            f"Melody conditioning: {'Enabled' if self.is_melody_model else 'Disabled'}"
        )

    def generate(
        self,
        prompt: str,
        duration: float = 30.0,
        guidance_scale: float = 3.0,
        temperature: float = 1.0,
        top_k: int = 250,
        top_p: float = 0.0,
        **kwargs,
    ) -> np.ndarray:
        """
        Generate music from text prompt using audiocraft.

        Args:
            prompt: Text description for music generation
            duration: Duration of generated music in seconds
            guidance_scale: Classifier-free guidance scale (cfg_coef in audiocraft)
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            **kwargs: Additional generation parameters

        Returns:
            Generated audio as numpy array (mono)
        """
        print(f"Generating music with prompt: '{prompt}'")
        print(
            f"Duration: {duration}s, Guidance scale: {guidance_scale}, Temperature: {temperature}"
        )

        # Set generation parameters
        self.model.set_generation_params(
            duration=duration,
            cfg_coef=guidance_scale,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        # Generate from text only
        wav = self.model.generate([prompt])

        # Convert to numpy (shape: [batch, channels, samples])
        audio = wav[0].cpu().numpy()

        # Convert to mono if stereo
        if audio.ndim == 2:
            audio = audio[0]  # Take first channel

        return audio

    def generate_with_melody(
        self,
        prompt: str,
        melody: np.ndarray,
        melody_sr: int = 32000,
        duration: float = 30.0,
        guidance_scale: float = 3.0,
        temperature: float = 1.0,
        **kwargs,
    ) -> np.ndarray:
        """
        Generate music with melody conditioning using audiocraft.

        Args:
            prompt: Text description for music generation
            melody: Melody audio as numpy array for conditioning
            melody_sr: Sample rate of melody audio
            duration: Duration of generated music in seconds
            guidance_scale: Classifier-free guidance scale
            temperature: Sampling temperature
            **kwargs: Additional generation parameters

        Returns:
            Generated audio as numpy array (mono)
        """
        if not self.is_melody_model:
            print(f"Warning: {self.model_name} doesn't support melody conditioning.")
            print("Falling back to text-only generation.")
            return self.generate(
                prompt, duration, guidance_scale, temperature, **kwargs
            )

        print(f"Generating music with prompt: '{prompt}' and melody conditioning")
        print(f"Duration: {duration}s, Guidance scale: {guidance_scale}")

        # Set generation parameters
        self.model.set_generation_params(
            duration=duration,
            cfg_coef=guidance_scale,
            temperature=temperature,
        )

        # Prepare melody for audiocraft
        # Convert numpy to torch tensor
        if isinstance(melody, np.ndarray):
            melody_tensor = torch.from_numpy(melody).float()
        else:
            melody_tensor = melody

        # Ensure melody is 2D (channels, samples)
        if melody_tensor.ndim == 1:
            melody_tensor = melody_tensor.unsqueeze(0)

        # Expand to batch size 1: (1, channels, samples)
        if melody_tensor.ndim == 2:
            melody_tensor = melody_tensor.unsqueeze(0)

        # Generate with melody conditioning using chroma features
        wav = self.model.generate_with_chroma(
            descriptions=[prompt],
            melody_wavs=melody_tensor,
            melody_sample_rate=melody_sr,
        )

        # Convert to numpy
        audio = wav[0].cpu().numpy()

        # Convert to mono if stereo
        if audio.ndim == 2:
            audio = audio[0]

        return audio

    def save_audio(self, audio: np.ndarray, output_path: str, sr: Optional[int] = None):
        """
        Save generated audio to file using audiocraft's audio_write.

        Args:
            audio: Audio array to save (mono, float32)
            output_path: Output file path
            sr: Sample rate (uses model's sample rate if not specified)
        """
        from audiocraft.data.audio import audio_write

        if sr is None:
            sr = self.sample_rate

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Remove .wav extension if present (audio_write adds it)
        output_name = str(output_path).replace(".wav", "")

        # Ensure audio is 2D (channels, samples) for audio_write
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]  # Add channel dimension

        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio).float()

        # Save using audiocraft's audio_write with loudness normalization
        audio_write(
            output_name, audio_tensor.cpu(), sr, strategy="loudness", format="wav"
        )

        print(f"Saved generated audio to: {output_path}")
