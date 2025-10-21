#!/usr/bin/env python3
"""Simple standalone text-to-music generator."""

import torch
import scipy.io.wavfile


def generate_music(
    prompt,
    output_file="output.wav",
    duration=30.0,
    cfg_scale=3.0,
    use_audiocraft=False,
    melody_path=None,
):
    """
    Generate music from text prompt.

    Args:
        prompt: Text description of music
        output_file: Output WAV file path
        duration: Duration in seconds (max 30)
        cfg_scale: Guidance scale (higher = more prompt adherence)
        use_audiocraft: Use audiocraft library instead of transformers
        melody_path: Path to melody audio file (only for audiocraft)
    """
    if use_audiocraft:
        # Use audiocraft library
        print(f"Loading MusicGen model using audiocraft...")
        from audiocraft.models import MusicGen
        from audiocraft.data.audio import audio_write

        model = MusicGen.get_pretrained('melody' if melody_path else 'medium')
        model.set_generation_params(
            duration=duration,
            cfg_coef=cfg_scale,
        )

        print(f"Model loaded! (Sample rate: {model.sample_rate} Hz)")
        print(f"\nPrompt: '{prompt}'")
        print(f"Generating {duration}s of music...")

        if melody_path:
            # Generate with melody conditioning
            import torchaudio
            print(f"Loading melody from: {melody_path}")
            melody, sr = torchaudio.load(melody_path)

            # Expand melody to batch size 1
            wav = model.generate_with_chroma([prompt], melody[None], sr)
        else:
            # Generate from text only
            wav = model.generate([prompt])

        # Save using audiocraft's audio_write (with loudness normalization)
        output_name = output_file.replace('.wav', '')
        audio_write(output_name, wav[0].cpu(), model.sample_rate, strategy="loudness")
        print(f"\n✓ Saved to: {output_name}.wav")

    else:
        # Use transformers library
        print(f"Loading MusicGen model using transformers...")
        from transformers import AutoProcessor, MusicgenForConditionalGeneration

        model_name = "facebook/musicgen-medium"
        processor = AutoProcessor.from_pretrained(model_name)
        model = MusicgenForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        sample_rate = model.config.audio_encoder.sampling_rate
        print(f"Model loaded! (Sample rate: {sample_rate} Hz)")

        # Generate
        print(f"\nPrompt: '{prompt}'")
        print(f"Generating {duration}s of music...")

        inputs = processor(text=[prompt], padding=True, return_tensors="pt")
        inputs = {k: v.to('cuda') for k, v in inputs.items()}

        with torch.no_grad():
            audio_values = model.generate(
                **inputs,
                max_new_tokens=int(duration * 50),
                guidance_scale=cfg_scale,
                do_sample=True,
            )

        # Save using scipy
        audio = audio_values[0, 0].cpu().numpy()
        scipy.io.wavfile.write(output_file, rate=sample_rate, data=audio)
        print(f"\n✓ Saved to: {output_file}")


if __name__ == "__main__":
    # Simple usage - just edit these values
    print("=" * 60)
    print("MusicGen - Text-to-Music Generator")
    print("=" * 60)

    prompt = input("Enter music description: ").strip()
    if not prompt:
        print("Error: Prompt cannot be empty!")
        exit(1)

    output = input("Output file (default: output.wav): ").strip() or "output.wav"

    use_audiocraft = input("Use audiocraft? (y/n, default: n): ").strip().lower() == 'y'

    melody_path = None
    if use_audiocraft:
        melody_input = input("Melody file path (optional, press Enter to skip): ").strip()
        if melody_input:
            melody_path = melody_input

    duration = input("Duration in seconds (default: 30): ").strip()
    duration = float(duration) if duration else 30.0

    cfg_scale = input("CFG scale (default: 3.0): ").strip()
    cfg_scale = float(cfg_scale) if cfg_scale else 3.0

    print("\n" + "=" * 60)
    generate_music(
        prompt,
        output_file=output,
        duration=duration,
        cfg_scale=cfg_scale,
        use_audiocraft=use_audiocraft,
        melody_path=melody_path,
    )
