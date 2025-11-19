import os
from pathlib import Path
from argparse import ArgumentParser, Namespace

import torch
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


def parse_arguments() -> Namespace:
    parser = ArgumentParser(description="Task 2")
    parser.add_argument(
        "--prompt_song_path",
        type=str,
        default="prompt_song",
        help="path of prompt song",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="checkpoints/11-08-23-02-35",
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
        help="number of target bars",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.2,
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.2,
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
        help="Specific epoch checkpoint to load (e.g., 10, 50). If None, uses latest checkpoint.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    # Determine checkpoint file and output folder based on epoch
    if args.epoch is not None:
        ckpt_file = f"checkpoint_epoch_{args.epoch}.pt"
        save_folder = Path(args.output_folder, f"{Path(args.ckpt_path).name}_epoch{args.epoch}", "task2")
    else:
        ckpt_file = CKPT_FILE
        save_folder = Path(args.output_folder, Path(args.ckpt_path).name, "task2")

    os.makedirs(save_folder, exist_ok=True)
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

    midi_paths = get_file_paths(args.prompt_song_path)
    truncated_midi_tokens = truncate_to_nbars(midi_paths, tokenizer, num_bar=8)

    device = get_device()
    model = AutoModelForCausalLM.from_pretrained(ckpt_config.model_name)
    model.load_state_dict(
        torch.load(Path(args.ckpt_path, ckpt_file), weights_only=True, map_location=device)["model"]
    )
    model.to(device)
    model.eval()

    generation_config = GenerationConfig(
        max_length=args.max_length,
        do_sample=True,
        top_k=args.top_k,
        temperature=args.temperature,
        pad_token_id=model.config.eos_token_id,
        repetition_penalty=args.repetition_penalty,
    )
    save_json(vars(args) | {"checkpoint": ckpt_config}, Path(save_folder, CONFIG_FILE))

    for i, data in enumerate(tqdm(truncated_midi_tokens), start=1):
        generated_tokens = generate_tokens(data, model, device, args.n_target_bar, BAR_TOKEN, generation_config)
        truncated_idx = get_trucated_idx(generated_tokens, tokenizer, args.n_target_bar)
        valid_tokens = filter_invalid_tokens(generated_tokens[:truncated_idx + 1], tokenizer)
        generated_midi = tokenizer.decode(valid_tokens)
        generated_midi.dump_midi(Path(save_folder, f"song_{i}.mid"))
