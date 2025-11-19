#!/usr/bin/env python3
"""
Task 2 with 3 different inference configurations.
Generates continuations for prompt songs and converts to WAV.
"""

import os
from pathlib import Path
from argparse import ArgumentParser, Namespace

import torch
import soundfile as sf
from midi2audio import FluidSynth
from miditok import REMI, TokenizerConfig
from transformers import AutoModelForCausalLM, GenerationConfig
from tqdm import tqdm

from src.constants import CONFIG_FILE, CKPT_FILE
from src.utils import (
    get_file_paths,
    get_device,
    get_trucated_idx,
    load_config,
    save_json,
    truncate_to_nbars,
    generate_tokens,
    filter_invalid_tokens,
)


# Define 3 different inference configurations
INFERENCE_CONFIGS = [
    {
        "name": "balanced",
        "top_k": 50,
        "temperature": 1.0,
        "repetition_penalty": 1.2,
    },
    {
        "name": "creative",
        "top_k": 100,
        "temperature": 1.5,
        "repetition_penalty": 1.1,
    },
    {
        "name": "conservative",
        "top_k": 20,
        "temperature": 0.8,
        "repetition_penalty": 1.3,
    },
]


def parse_arguments() -> Namespace:
    parser = ArgumentParser(description="Task 2 with 3 inference configurations")
    parser.add_argument(
        "--prompt_song_path",
        type=str,
        default="prompt_song",
        help="path of prompt song",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="path of checkpoint",
    )
    parser.add_argument(
        "--num_velocities",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--n_target_bar",
        type=int,
        default=32,
        help="number of target bars (8 prompt + 24 continuation)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="results",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="Specific epoch checkpoint to load",
    )
    return parser.parse_args()


def midi_to_wav(midi_path: Path, wav_path: Path, soundfont: Path):
    """Convert MIDI file to WAV using FluidSynth."""
    fs = FluidSynth(str(soundfont))
    fs.midi_to_audio(str(midi_path), str(wav_path))


def main():
    args = parse_arguments()
    device = get_device()

    # Determine checkpoint file
    if args.epoch is not None:
        ckpt_file = f"checkpoint_epoch_{args.epoch}.pt"
        base_folder = Path(args.output_folder, f"{Path(args.ckpt_path).name}_epoch{args.epoch}", "task2_3sets")
    else:
        ckpt_file = CKPT_FILE
        base_folder = Path(args.output_folder, Path(args.ckpt_path).name, "task2_3sets")

    # Setup soundfont
    script_dir = Path(__file__).parent.parent
    soundfont_file = script_dir / "soundfonts" / "Dore Mark's NY S&S Model B-v5.2.sf2"

    # Load config and tokenizer
    ckpt_config = load_config(Path(args.ckpt_path, CONFIG_FILE))
    tokenizer_config = TokenizerConfig(
        num_velocities=args.num_velocities,
        use_chords=True,
        use_tempos=True,
        use_programs=True,
        params=Path(args.ckpt_path, "tokenizer.json")
    )
    tokenizer = REMI(tokenizer_config)
    BAR_TOKEN = [v for k, v in tokenizer.vocab.items() if "Bar" in k][0]

    # Load prompt songs
    midi_paths = get_file_paths(args.prompt_song_path)
    truncated_midi_tokens = truncate_to_nbars(midi_paths, tokenizer, num_bar=8)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(ckpt_config.model_name)
    model.load_state_dict(
        torch.load(Path(args.ckpt_path, ckpt_file), weights_only=True, map_location=device)["model"]
    )
    model.to(device)
    model.eval()

    # Process each configuration
    for config in INFERENCE_CONFIGS:
        config_name = config["name"]
        save_folder = base_folder / config_name
        wav_folder = base_folder / f"{config_name}_wav"
        os.makedirs(save_folder, exist_ok=True)
        os.makedirs(wav_folder, exist_ok=True)

        print(f"\n{'='*50}")
        print(f"Configuration: {config_name}")
        print(f"  top_k: {config['top_k']}, temp: {config['temperature']}, rep_penalty: {config['repetition_penalty']}")
        print(f"{'='*50}")

        generation_config = GenerationConfig(
            max_length=args.max_length,
            do_sample=True,
            top_k=config["top_k"],
            temperature=config["temperature"],
            pad_token_id=model.config.eos_token_id,
            repetition_penalty=config["repetition_penalty"],
        )

        # Save config
        save_config = vars(args).copy()
        save_config.update({
            "checkpoint": ckpt_config,
            "inference_config": config,
        })
        save_json(save_config, Path(save_folder, CONFIG_FILE))

        # Generate for each prompt song
        for i, data in enumerate(tqdm(truncated_midi_tokens, desc=f"Generating ({config_name})"), start=1):
            # Generate MIDI
            generated_tokens = generate_tokens(data, model, device, args.n_target_bar, BAR_TOKEN, generation_config)
            truncated_idx = get_trucated_idx(generated_tokens, tokenizer, args.n_target_bar)
            valid_tokens = filter_invalid_tokens(generated_tokens[:truncated_idx + 1], tokenizer)
            generated_midi = tokenizer.decode(valid_tokens)

            midi_path = Path(save_folder, f"song_{i}.mid")
            generated_midi.dump_midi(midi_path)

            # Convert to WAV
            wav_path = Path(wav_folder, f"song_{i}.wav")
            midi_to_wav(midi_path, wav_path, soundfont_file)

        print(f"Saved MIDI files to: {save_folder}")
        print(f"Saved WAV files to: {wav_folder}")

    print(f"\nAll configurations completed! Results in: {base_folder}")


if __name__ == "__main__":
    main()
