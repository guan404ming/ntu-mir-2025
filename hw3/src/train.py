import os
from argparse import ArgumentParser, Namespace
from pathlib import Path

import wandb
from miditok import REMI, TokenizerConfig
from miditok.pytorch_data import DatasetMIDI, DataCollator
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

from src.constants import PROJECT_NAME, CHECKPOINT_DIR, CONFIG_FILE
from src.optimizer import get_optimizer
from src.lr_scheduler import get_lr_scheduler
from src.trainer import Trainer
from src.utils import (
    set_random_seeds,
    get_device,
    get_time,
    save_json,
)

os.environ["WANDB_MODE"] = "offline"

def parse_arguments() -> Namespace:
    parser = ArgumentParser(description="Symbolic Music Generation")

    # dataset setting
    parser.add_argument(
        "--data_folder",
        type=str,
        default="Pop1K7/midi_analyzed",
        help="folder of dataset"
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="remi",
        choices=["remi", "remiplus", "midilike"],
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for dataloader.",
    )

    # training setting
    parser.add_argument(
        "--model_name",
        type=str,
        default="BabyLM-community/babylm-baseline-100m-gpt2",
        choices=[
            "BabyLM-community/babylm-baseline-100m-gpt2",
            "gpt2",
            "EleutherAI/gpt-neo-125M",
        ],
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["sgd", "adam", "adamw"],
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=4e-4,
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5,
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="one_cycle",
        choices=["constant", "step", "one_cycle", "cosine_annealing"],
    )
    return parser.parse_args()


if __name__ == "__main__":
    set_random_seeds()
    args = parse_arguments()
    checkpoint_dir = Path(CHECKPOINT_DIR, get_time())
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_json(vars(args), Path(checkpoint_dir, CONFIG_FILE))

    config = TokenizerConfig(
        num_velocities=16,
        use_chords=True,
        use_programs=True,
        use_tempos=True,
        params=Path("tokenizers", f"{args.tokenizer_name}.json").stem
    )
    tokenizer = REMI(config)
    tokenizer.save(Path(checkpoint_dir, "tokenizer.json"))
    dataset = DatasetMIDI(
        files_paths=list(Path(args.data_folder).glob("**/*.mid")),
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        bos_token_id=tokenizer["BOS_None"],
        eos_token_id=tokenizer["EOS_None"],
    )
    collator = DataCollator(
        tokenizer.pad_token_id,
        copy_inputs_as_labels=True,
    )
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collator,
    )

    # Prepare model
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    device = get_device()
    optimizer = get_optimizer(
        name=args.optimizer,
        model=model,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    lr_scheduler = get_lr_scheduler(
        name=args.lr_scheduler,
        optimizer=optimizer,
        max_lr=args.lr,
        steps_for_one_epoch=len(train_loader),
        epochs=args.epochs,
    )

    # Prepare logger
    wandb.init(
        project=PROJECT_NAME,
        name=os.path.basename(checkpoint_dir),
        config=vars(args),
    )
    wandb.watch(model, log="all")

    trainer = Trainer(
        model=model,
        device=device,
        train_loader=train_loader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accum_grad_step=1,
        clip_grad_norm=1.0,
        logger=wandb,
        checkpoint_dir=checkpoint_dir,
    )
    trainer.fit(epochs=args.epochs)
