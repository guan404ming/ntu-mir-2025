"""MusicGen-based music generator using transformers (no spacy dependency)."""

import torch
import numpy as np
import scipy.io.wavfile
import scipy.signal
from pathlib import Path
from typing import Optional
from .base_generator import BaseGenerator


class MusicGenGenerator(BaseGenerator):
    """Music generator using Meta's MusicGen model via transformers."""

    def __init__(
        self, model_name: str = "facebook/musicgen-medium", device: str = "cuda"
    ):
        """
        Initialize MusicGen generator using transformers library.

        Args:
            model_name: Hugging Face model name (musicgen-small/medium/large/melody)
            device: Device to use (cuda/cpu)
        """
        self.device = device
        self.model_name = model_name

        print(f"Loading MusicGen model via transformers: {model_name}")

        # Check if this is the melody-conditioned model
        self.is_melody_model = "melody" in model_name.lower()

        # Import the appropriate model class
        if self.is_melody_model:
            from transformers import (
                AutoProcessor,
                MusicgenMelodyForConditionalGeneration,
            )

            model_class = MusicgenMelodyForConditionalGeneration
        else:
            from transformers import AutoProcessor, MusicgenForConditionalGeneration

            model_class = MusicgenForConditionalGeneration

        # Load processor and model
        self.processor = AutoProcessor.from_pretrained(model_name)

        # Use device_map="auto" for CUDA to manage memory efficiently
        if device == "cuda":
            self.model = model_class.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
            self.dtype = torch.float16
        else:
            self.model = model_class.from_pretrained(
                model_name, torch_dtype=torch.float32
            )
            self.model.to(device)
            self.dtype = torch.float32

        self.model.eval()

        self.sample_rate = self.model.config.audio_encoder.sampling_rate

        print(f"MusicGen model loaded (SR: {self.sample_rate} Hz)")
        print(f"Model dtype: {self.dtype}")
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
        do_sample: bool = True,
        **kwargs,
    ) -> np.ndarray:
        """
        Generate music from text prompt following official Transformers usage.

        Args:
            prompt: Text description for music generation
            duration: Duration of generated music in seconds (max 30s)
            guidance_scale: Classifier-free guidance scale (default=3.0, higher = more prompt adherence)
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            do_sample: Whether to use sampling (recommended over greedy decoding)
            **kwargs: Additional generation parameters

        Returns:
            Generated audio as numpy array (mono)
        """
        print(f"Generating music with prompt: '{prompt}'")
        print(
            f"Duration: {duration}s, Guidance scale: {guidance_scale}, Temperature: {temperature}"
        )

        # Process inputs following official Transformers usage
        inputs = self.processor(
            text=[prompt],
            padding=True,
            return_tensors="pt",
        )

        # Move inputs to device with correct dtype
        if self.device == "cuda":
            inputs = {
                k: v.to(
                    self.device,
                    dtype=self.dtype if v.dtype.is_floating_point else v.dtype,
                )
                for k, v in inputs.items()
            }

        # Calculate max_new_tokens from duration
        # MusicGen generates at ~50 tokens/second, max 1503 tokens (30 seconds)
        max_new_tokens = min(int(duration * 50), 1503)

        # Generate following official API
        with torch.no_grad():
            audio_values = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                guidance_scale=guidance_scale,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p if top_p > 0 else None,
            )

        # Convert to numpy (shape: [batch, channels, samples])
        audio = audio_values.cpu().numpy()[0, 0]  # Get first batch, first channel

        return audio

    def generate_with_melody(
        self,
        prompt: str,
        melody: np.ndarray,
        melody_sr: int = 32000,
        duration: float = 30.0,
        guidance_scale: float = 3.0,
        temperature: float = 1.0,
        do_sample: bool = True,
        **kwargs,
    ) -> np.ndarray:
        """
        Generate music with melody conditioning.

        Args:
            prompt: Text description for music generation
            melody: Melody audio as numpy array for conditioning
            melody_sr: Sample rate of melody audio
            duration: Duration of generated music in seconds
            guidance_scale: Classifier-free guidance scale
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            **kwargs: Additional generation parameters

        Returns:
            Generated audio as numpy array (mono)
        """
        if not self.is_melody_model:
            print(f"Warning: {self.model_name} doesn't support melody conditioning.")
            print("Falling back to text-only generation.")
            return self.generate(
                prompt, duration, guidance_scale, temperature, do_sample, **kwargs
            )

        print(f"Generating music with prompt: '{prompt}' and melody conditioning")
        print(f"Duration: {duration}s, Guidance scale: {guidance_scale}")

        # Prepare melody array - processor expects numpy array
        melody_array = melody.copy()

        # Resample melody if needed
        if melody_sr != self.sample_rate:
            # Use scipy for resampling
            if melody_array.ndim == 1:
                # Mono audio
                num_samples = int(len(melody_array) * self.sample_rate / melody_sr)
                melody_array = scipy.signal.resample(melody_array, num_samples)
            else:
                # Multi-channel audio - resample first channel only
                num_samples = int(melody_array.shape[1] * self.sample_rate / melody_sr)
                melody_array = scipy.signal.resample(melody_array[0], num_samples)

        # Process inputs with both text and audio (processor expects numpy array)
        inputs = self.processor(
            text=[prompt],
            audio=melody_array,
            sampling_rate=self.sample_rate,
            padding=True,
            return_tensors="pt",
        )

        # Move inputs to device with correct dtype
        if self.device == "cuda":
            inputs = {
                k: v.to(
                    self.device,
                    dtype=self.dtype if v.dtype.is_floating_point else v.dtype,
                )
                for k, v in inputs.items()
            }

        # Calculate max_new_tokens from duration
        max_new_tokens = min(int(duration * 50), 1503)

        # Generate with melody conditioning
        with torch.no_grad():
            audio_values = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                guidance_scale=guidance_scale,
                do_sample=do_sample,
                temperature=temperature,
            )

        # Convert to numpy
        audio = audio_values.cpu().numpy()[0, 0]

        return audio

    def save_audio(self, audio: np.ndarray, output_path: str, sr: Optional[int] = None):
        """
        Save generated audio to file using scipy.

        Args:
            audio: Audio array to save (mono, float32)
            output_path: Output file path
            sr: Sample rate (uses model's sample rate if not specified)
        """
        if sr is None:
            sr = self.sample_rate

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Ensure audio is 1D (mono)
        if audio.ndim == 2:
            audio = audio[0]  # Take first channel if stereo

        # # Normalize audio to prevent clipping (values outside [-1, 1])
        # # This is crucial for quality - MusicGen sometimes produces values > 1.0
        # max_abs = np.abs(audio).max()
        # if max_abs > 1.0:
        #     print(f"  Normalizing audio (max: {max_abs:.2f}) to prevent clipping")
        #     audio = audio / max_abs * 0.95  # Scale to 95% to leave some headroom

        # scipy.io.wavfile.write expects float32 in range [-1, 1] or int16
        # We'll keep it as float32 for best quality
        audio = audio.astype(np.float32)

        # Save using scipy
        scipy.io.wavfile.write(str(output_path), rate=sr, data=audio)
        print(f"Saved generated audio to: {output_path}")
