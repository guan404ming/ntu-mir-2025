"""MusicGen-based music generator using transformers (no spacy dependency)."""

import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Optional
from .base_generator import BaseGenerator


class MusicGenGenerator(BaseGenerator):
    """Music generator using Meta's MusicGen model via transformers."""

    def __init__(
        self,
        model_name: str = "facebook/musicgen-medium",
        device: str = "cuda"
    ):
        """
        Initialize MusicGen generator using transformers library.

        Args:
            model_name: Hugging Face model name (musicgen-small/medium/large/melody)
            device: Device to use (cuda/cpu)
        """
        from transformers import AutoProcessor, MusicgenForConditionalGeneration

        self.device = device
        self.model_name = model_name

        print(f"Loading MusicGen model via transformers: {model_name}")

        # Load processor and model
        self.processor = AutoProcessor.from_pretrained(model_name)

        # Use device_map="auto" for CUDA to manage memory efficiently
        if device == "cuda":
            self.model = MusicgenForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
        else:
            self.model = MusicgenForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float32
            )
            self.model.to(device)

        self.model.eval()

        self.sample_rate = self.model.config.audio_encoder.sampling_rate

        # Check if this is the melody-conditioned model
        self.is_melody_model = "melody" in model_name.lower()

        print(f"MusicGen model loaded (SR: {self.sample_rate} Hz)")
        print(f"Melody conditioning: {'Enabled' if self.is_melody_model else 'Disabled'}")

    def generate(
        self,
        prompt: str,
        duration: float = 30.0,
        cfg_scale: float = 3.0,
        temperature: float = 1.0,
        top_k: int = 250,
        top_p: float = 0.0,
        **kwargs
    ) -> np.ndarray:
        """
        Generate music from text prompt only.

        Args:
            prompt: Text description for music generation
            duration: Duration of generated music in seconds
            cfg_scale: Classifier-free guidance scale (higher = more prompt adherence)
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            **kwargs: Additional generation parameters

        Returns:
            Generated audio as numpy array (mono)
        """
        print(f"Generating music with prompt: '{prompt}'")
        print(f"Duration: {duration}s, CFG scale: {cfg_scale}, Temperature: {temperature}")

        # Process inputs
        inputs = self.processor(
            text=[prompt],
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        # Calculate max_new_tokens from duration
        # MusicGen generates at ~50 tokens/second
        max_new_tokens = int(duration * 50)

        # Generate
        with torch.no_grad():
            audio_values = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                guidance_scale=cfg_scale,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p if top_p > 0 else None,
                do_sample=True,
            )

        # Convert to numpy (shape: [batch, channels, samples])
        audio = audio_values.cpu().numpy()[0, 0]  # Get first batch, first channel

        return audio

    def generate_with_melody(
        self,
        prompt: str,
        melody: np.ndarray,
        melody_sr: int = 24000,
        duration: float = 30.0,
        cfg_scale: float = 3.0,
        temperature: float = 1.0,
        **kwargs
    ) -> np.ndarray:
        """
        Generate music with melody conditioning.

        Args:
            prompt: Text description for music generation
            melody: Melody audio as numpy array for conditioning
            melody_sr: Sample rate of melody audio
            duration: Duration of generated music in seconds
            cfg_scale: Classifier-free guidance scale
            temperature: Sampling temperature
            **kwargs: Additional generation parameters

        Returns:
            Generated audio as numpy array (mono)
        """
        if not self.is_melody_model:
            print(f"Warning: {self.model_name} doesn't support melody conditioning.")
            print("Falling back to text-only generation.")
            return self.generate(prompt, duration, cfg_scale, temperature, **kwargs)

        print(f"Generating music with prompt: '{prompt}' and melody conditioning")
        print(f"Duration: {duration}s, CFG scale: {cfg_scale}")

        # Prepare melody tensor
        if melody.ndim == 1:
            melody = melody[np.newaxis, :]  # Add channel dimension

        melody_tensor = torch.from_numpy(melody).float()

        # Resample melody if needed
        if melody_sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(melody_sr, self.sample_rate)
            melody_tensor = resampler(melody_tensor)

        # Ensure correct shape [1, channels, samples]
        if melody_tensor.ndim == 2:
            melody_tensor = melody_tensor.unsqueeze(0)  # Add batch dimension

        melody_tensor = melody_tensor.to(self.device)

        # Note: The current transformers implementation of MusicGen doesn't fully support
        # melody conditioning via the generate() method. The processor returns 'input_features'
        # but generate() doesn't accept it. For now, we fall back to text-only generation.
        # TODO: Implement melody conditioning using audiocraft library or wait for transformers fix

        print("Warning: Melody conditioning not fully supported in transformers implementation.")
        print("Falling back to text-only generation with enhanced prompt.")

        # Fall back to text-only generation
        return self.generate(prompt, duration, cfg_scale, temperature, **kwargs)

    def save_audio(
        self,
        audio: np.ndarray,
        output_path: str,
        sr: Optional[int] = None
    ):
        """
        Save generated audio to file.

        Args:
            audio: Audio array to save
            output_path: Output file path
            sr: Sample rate (uses model's sample rate if not specified)
        """
        if sr is None:
            sr = self.sample_rate

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Ensure correct shape
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]  # Add channel dimension

        # Normalize audio to prevent clipping (values outside [-1, 1])
        # This is crucial for quality - MusicGen sometimes produces values > 1.0
        max_abs = np.abs(audio).max()
        if max_abs > 1.0:
            print(f"  Normalizing audio (max: {max_abs:.2f}) to prevent clipping")
            audio = audio / max_abs * 0.95  # Scale to 95% to leave some headroom

        # Convert to tensor
        audio_tensor = torch.from_numpy(audio).float()

        # Save
        torchaudio.save(str(output_path), audio_tensor, sr)
        print(f"Saved generated audio to: {output_path}")
