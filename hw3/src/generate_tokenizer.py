from pathlib import Path
from argparse import ArgumentParser, Namespace

from miditok import TokenizerConfig, REMI, MIDILike


def parse_arguments() -> Namespace:
    parser = ArgumentParser(description="Tokenizer Generation")
    parser.add_argument(
        "--data_folder",
        type=str,
        default="Pop1K7/midi_analyzed",
        help="folder of dataset"
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="remi",
        choices=["remi", "remiplus", "midilike"],
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    match args.tokenizer_name:
        case "remi":
            config = TokenizerConfig(
                num_velocities=16,
                use_chords=True,
                use_programs=False,
                use_tempos=True,
            )
            tokenizer = REMI(config)
        case "remiplus":
            config = TokenizerConfig(
                num_velocities=16,
                use_chords=True,
                use_programs=False,
                use_tempos=True,
                one_token_stream_for_programs = True,
                use_time_signatures = True,
            )
            tokenizer = REMI(config)
        case "midilike":
            config = TokenizerConfig(
                num_velocities=16,
                use_chords=True,
                use_programs=False,
                use_tempos=True,
                max_duration=(4, 480, 120)
            )
            tokenizer = MIDILike(config)
        case _:
            raise ValueError("Invalid tokenizer name")

    tokenizer.train(
        vocab_size=30000,
        files_paths=list(Path(args.data_folder).glob("**/*.mid")),
    )
    tokenizer.save(Path("tokenizers", f"{args.tokenizer_name}.json"))
