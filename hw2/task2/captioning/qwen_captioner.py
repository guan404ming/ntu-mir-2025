"""Qwen-Audio based music captioner."""

import torch
from typing import Dict, Any
from pathlib import Path
from .base_captioner import BaseCaptioner


class QwenAudioCaptioner(BaseCaptioner):
    """Music captioner using Qwen-Audio model."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-Audio-7B-Instruct",
        device: str = "cuda",
        use_8bit: bool = False,
    ):
        """
        Initialize Qwen-Audio captioner.

        Args:
            model_name: Hugging Face model name
            device: Device to use (cuda/cpu)
            use_8bit: Use 8-bit quantization to reduce memory (requires bitsandbytes)
        """
        from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
        import warnings

        self.device = device
        self.model_name = model_name

        print(f"Loading Qwen-Audio model: {model_name}")

        # Suppress the WhisperFeatureExtractor sampling_rate warning
        # This is a known issue in transformers where AutoProcessor doesn't pass
        # sampling_rate to WhisperFeatureExtractor, but it's harmless
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="It is strongly recommended to pass the `sampling_rate` argument",
            )
            self.processor = AutoProcessor.from_pretrained(model_name)

        # Prepare model loading arguments
        model_kwargs = {
            "low_cpu_mem_usage": True,
        }

        # Use dtype and device_map based on device and quantization
        if device == "cuda":
            if use_8bit:
                # Use 8-bit quantization to save ~50% memory
                print("  Using 8-bit quantization to reduce memory usage")
                model_kwargs["load_in_8bit"] = True
                model_kwargs["device_map"] = "auto"
            else:
                model_kwargs["torch_dtype"] = torch.float16
                model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["torch_dtype"] = torch.float32

        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_name, **model_kwargs
        )

        if device == "cpu" and not use_8bit:
            self.model = self.model.to(device)

        self.model.eval()

        print("Qwen-Audio model loaded successfully")
        if device == "cuda":
            if use_8bit:
                print("  Using 8-bit quantization (saves ~50% memory)")
            else:
                print(f"  Using {model_kwargs.get('torch_dtype', 'default')} precision")

    def unload(self):
        """Unload model from memory to free GPU."""
        import gc

        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "processor"):
            del self.processor

        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
            print("Qwen-Audio model unloaded from GPU")

    def generate_caption(
        self,
        audio_path: str,
        prompt: str = "Describe this music in detail, including genre, instruments, mood, tempo, and key musical elements.",
        max_new_tokens: int = 256,
        **kwargs,
    ) -> str:
        """
        Generate a text caption describing the music.

        Args:
            audio_path: Path to the audio file
            prompt: Instruction prompt for caption generation
            max_new_tokens: Maximum number of tokens to generate
            **kwargs: Additional generation parameters

        Returns:
            Generated text caption
        """
        import librosa

        # Load audio at the correct sampling rate
        audio_data, _ = librosa.load(
            audio_path, sr=self.processor.feature_extractor.sampling_rate
        )

        # Prepare conversation format
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio_url": audio_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Apply chat template
        text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )

        # Process inputs with loaded audio data
        inputs = self.processor(
            text=text, audio=audio_data, return_tensors="pt", padding=True
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            generate_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
            )

        # Remove input tokens and decode
        generate_ids = generate_ids[:, inputs["input_ids"].size(1) :]
        caption = self.processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        # Clear GPU memory
        if self.device == "cuda":
            del inputs, generate_ids
            torch.cuda.empty_cache()

        return caption.strip()

    def generate_detailed_caption(self, audio_path: str) -> Dict[str, Any]:
        """
        Generate detailed caption with multiple aspects using Qwen-Audio model.

        Args:
            audio_path: Path to the audio file

        Returns:
            Dictionary with caption and metadata
        """
        # Use a single comprehensive prompt to get all information at once
        # This is more memory efficient than multiple separate generations
        main_prompt = """Describe this music in detail like a music critic, including:
1. Genre
2. Instruments
3. Mood and emotional character
4. Key musical elements

Provide a comprehensive description."""

        main_caption = self.generate_caption(
            audio_path, prompt=main_prompt, max_new_tokens=256
        )

        # Parse the main caption to extract individual components
        # For now, return the full caption for all fields
        # The main_caption contains all the information
        return {
            "audio_path": str(audio_path),
            "file_name": Path(audio_path).name,
            "main_caption": main_caption,
        }
