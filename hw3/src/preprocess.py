"""
Data preprocessing script for Pop1K7 dataset using REMI+ tokenization.
"""
import os
import pickle
from pathlib import Path
from typing import List, Dict
import numpy as np
from miditok import REMI
from symusic import Score
from tqdm import tqdm
import argparse


def create_tokenizer(vocab_size_limit: int = None) -> REMI:
    """
    Create REMI+ tokenizer with appropriate configuration for piano music.

    REMI+ uses:
    - Bar events
    - Position events (for timing within bars)
    - Pitch events (MIDI note numbers)
    - Velocity events (note dynamics)
    - Duration events (note length)
    - Tempo events
    """
    from miditok import TokenizerConfig

    config = TokenizerConfig(
        pitch_range=(21, 109),  # Piano range: A0 to C8
        beat_res={(0, 4): 8},   # 8 positions per beat (32 per bar for 4/4)
        num_velocities=32,      # 32 velocity bins
        use_chords=False,       # Not using chord events
        use_rests=False,        # Not using rest events
        use_tempos=True,        # Use tempo events
        use_time_signatures=False,  # Dataset is all 4/4
        use_programs=False,     # Single instrument (piano)
        num_tempos=32,          # 32 tempo bins
    )

    tokenizer = REMI(tokenizer_config=config)
    return tokenizer


def preprocess_dataset(
    data_dir: str = "data/Pop1K7/midi_analyzed",
    output_dir: str = "data/processed",
    tokenizer_path: str = "data/processed/tokenizer.pkl",
    max_bars: int = 32,
    create_vocab: bool = True
):
    """
    Preprocess the Pop1K7 dataset.

    Args:
        data_dir: Directory containing MIDI files
        output_dir: Directory to save processed data
        tokenizer_path: Path to save/load tokenizer
        max_bars: Maximum number of bars to process per segment
        create_vocab: Whether to create vocabulary from scratch
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get all MIDI files (recursively search subdirectories)
    midi_files = list(Path(data_dir).rglob("*.mid")) + list(Path(data_dir).rglob("*.midi"))
    print(f"Found {len(midi_files)} MIDI files")

    if create_vocab:
        print("Creating tokenizer and building vocabulary...")
        tokenizer = create_tokenizer()

        # REMI tokenizer has predefined vocabulary based on config
        # No need to train vocabulary like BPE-based tokenizers

        # Save tokenizer
        save_path = Path(tokenizer_path).with_suffix('.json')
        tokenizer.save_pretrained(save_path.parent, filename=save_path.name)
        print(f"Tokenizer saved to {save_path}")
        print(f"Vocabulary size: {len(tokenizer)}")
    else:
        # Load existing tokenizer
        load_path = Path(tokenizer_path).with_suffix('.json')
        tokenizer = REMI.from_pretrained(load_path.parent, filename=load_path.name)
        print(f"Loaded tokenizer with vocabulary size: {len(tokenizer)}")

    # Process all MIDI files and tokenize
    all_tokens = []
    failed_files = []

    for midi_file in tqdm(midi_files, desc="Tokenizing MIDI files"):
        try:
            # Load MIDI file
            score = Score(str(midi_file))

            # Tokenize - returns TokSequence objects
            tokens = tokenizer(score)

            # Convert to token IDs
            if hasattr(tokens, 'ids'):
                # Single track
                token_ids = tokens.ids
            elif isinstance(tokens, list) and len(tokens) > 0:
                # Multi-track - take first track
                if hasattr(tokens[0], 'ids'):
                    token_ids = tokens[0].ids
                else:
                    token_ids = tokens[0]
            else:
                continue

            if len(token_ids) > 0:
                all_tokens.append({
                    'file': str(midi_file.name),
                    'tokens': token_ids
                })

        except Exception as e:
            print(f"Error processing {midi_file.name}: {e}")
            failed_files.append(str(midi_file.name))
            continue

    print(f"Successfully tokenized {len(all_tokens)} files")
    print(f"Failed to tokenize {len(failed_files)} files")

    # Save tokenized data
    output_file = Path(output_dir) / "tokenized_data.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(all_tokens, f)
    print(f"Saved tokenized data to {output_file}")

    # Create segments for training (each segment is max_bars)
    # We need to identify Bar tokens to segment properly
    print(f"\nCreating training segments ({max_bars} bars each)...")
    segments = create_segments(all_tokens, tokenizer, max_bars)

    # Save segments
    segments_file = Path(output_dir) / f"segments_{max_bars}bars.pkl"
    with open(segments_file, 'wb') as f:
        pickle.dump(segments, f)
    print(f"Saved {len(segments)} segments to {segments_file}")

    # Print statistics
    print_statistics(segments, tokenizer)

    return tokenizer, all_tokens, segments


def create_segments(all_tokens: List[Dict], tokenizer: REMI, max_bars: int = 32) -> List[np.ndarray]:
    """
    Create fixed-length segments from tokenized data.
    Each segment contains max_bars of music.
    """
    segments = []

    # Find Bar token ID - REMI uses event structure
    bar_token_id = None
    for token_str, token_id in tokenizer.vocab.items():
        if token_str.startswith('Bar'):
            bar_token_id = token_id
            print(f"Found Bar token: '{token_str}' with ID: {bar_token_id}")
            break

    if bar_token_id is None:
        print("Warning: Could not find Bar token, using fixed-length segmentation")
        # Fallback to fixed-length segments
        for item in tqdm(all_tokens, desc="Creating fixed segments"):
            tokens = item['tokens']
            segment_length = 512  # Arbitrary length
            for i in range(0, len(tokens) - segment_length, segment_length // 2):
                segment = tokens[i:i + segment_length]
                if len(segment) == segment_length:
                    segments.append(np.array(segment))
    else:
        # Segment by bars
        for item in tqdm(all_tokens, desc="Creating bar segments"):
            tokens = item['tokens']

            # Find bar positions
            bar_positions = [i for i, t in enumerate(tokens) if t == bar_token_id]

            if len(bar_positions) < max_bars + 1:  # Need max_bars+1 to create segments
                continue

            # Create segments of max_bars
            for i in range(len(bar_positions) - max_bars):
                start_pos = bar_positions[i]
                end_pos = bar_positions[i + max_bars]

                segment = tokens[start_pos:end_pos]
                if len(segment) > 0:
                    segments.append(np.array(segment))

    return segments


def print_statistics(segments: List[np.ndarray], tokenizer: REMI):
    """Print statistics about the processed data."""
    if len(segments) == 0:
        print("No segments created!")
        return

    segment_lengths = [len(seg) for seg in segments]
    print("\n=== Dataset Statistics ===")
    print(f"Total segments: {len(segments)}")
    print(f"Vocabulary size: {len(tokenizer)}")
    print(f"Segment length - Mean: {np.mean(segment_lengths):.1f}, "
          f"Std: {np.std(segment_lengths):.1f}, "
          f"Min: {np.min(segment_lengths)}, "
          f"Max: {np.max(segment_lengths)}")

    # Token distribution
    all_tokens_flat = np.concatenate(segments)
    unique_tokens, counts = np.unique(all_tokens_flat, return_counts=True)
    print(f"Unique tokens used: {len(unique_tokens)} / {len(tokenizer)}")
    print(f"Total tokens: {len(all_tokens_flat)}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess Pop1K7 dataset with REMI+ tokenization")
    parser.add_argument("--data_dir", type=str, default="data/Pop1K7/midi_analyzed",
                        help="Directory containing MIDI files")
    parser.add_argument("--output_dir", type=str, default="data/processed",
                        help="Directory to save processed data")
    parser.add_argument("--max_bars", type=int, default=32,
                        help="Maximum number of bars per segment")
    parser.add_argument("--create_vocab", action="store_true", default=True,
                        help="Create vocabulary from scratch")

    args = parser.parse_args()

    tokenizer, all_tokens, segments = preprocess_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        tokenizer_path=os.path.join(args.output_dir, "tokenizer.pkl"),
        max_bars=args.max_bars,
        create_vocab=args.create_vocab
    )

    print("\nPreprocessing complete!")


if __name__ == "__main__":
    main()
