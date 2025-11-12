"""
Dataset class for music generation training.
"""
import pickle
import random
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset


class MusicDataset(Dataset):
    """
    Dataset for autoregressive music generation.

    Each sample returns (input_sequence, target_sequence) where:
    - input_sequence: tokens [0:seq_len]
    - target_sequence: tokens [1:seq_len+1] (shifted by 1 for next-token prediction)
    """

    def __init__(
        self,
        segments_path: str,
        seq_len: int = 512,
        stride: int = None,
        augment: bool = True,
        pitch_shift_range: int = 6,
    ):
        """
        Args:
            segments_path: Path to preprocessed segments pickle file
            seq_len: Length of sequence for training
            stride: Stride for creating windows from segments (default: seq_len)
            augment: Whether to apply data augmentation
            pitch_shift_range: Range for pitch shifting augmentation (in semitones)
        """
        self.seq_len = seq_len
        self.stride = stride if stride is not None else seq_len
        self.augment = augment
        self.pitch_shift_range = pitch_shift_range

        # Load segments
        with open(segments_path, 'rb') as f:
            self.segments = pickle.load(f)

        print(f"Loaded {len(self.segments)} segments")

        # Create windows from segments
        self.windows = []
        self._create_windows()

        print(f"Created {len(self.windows)} training windows")

    def _create_windows(self):
        """Create fixed-length windows from variable-length segments."""
        for seg_idx, segment in enumerate(self.segments):
            if len(segment) < self.seq_len + 1:
                # Segment too short, skip
                continue

            # Create overlapping windows with stride
            for start_idx in range(0, len(segment) - self.seq_len - 1, self.stride):
                end_idx = start_idx + self.seq_len + 1
                if end_idx <= len(segment):
                    self.windows.append({
                        'segment_idx': seg_idx,
                        'start': start_idx,
                        'end': end_idx
                    })

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        """Get a training sample (input_seq, target_seq)."""
        window = self.windows[idx]
        segment = self.segments[window['segment_idx']]
        tokens = segment[window['start']:window['end']]

        # Apply augmentation if enabled
        if self.augment and random.random() < 0.5:
            tokens = self._augment_pitch_shift(tokens)

        # Split into input and target
        input_seq = tokens[:-1]
        target_seq = tokens[1:]

        # Convert to tensors
        input_tensor = torch.tensor(input_seq, dtype=torch.long)
        target_tensor = torch.tensor(target_seq, dtype=torch.long)

        return input_tensor, target_tensor

    def _augment_pitch_shift(self, tokens: np.ndarray) -> np.ndarray:
        """
        Apply pitch shifting augmentation.
        This is a simplified version - ideally should respect token vocabulary structure.
        """
        # For proper implementation, need to identify pitch tokens and shift them
        # This is a placeholder - implement based on tokenizer structure
        # For now, return unchanged
        return tokens


class ConditionalMusicDataset(Dataset):
    """
    Dataset for conditional music generation (continuation).
    Returns (prompt_tokens, continuation_tokens).
    """

    def __init__(
        self,
        segments_path: str,
        prompt_bars: int = 8,
        continuation_bars: int = 24,
        seq_len: int = 512
    ):
        """
        Args:
            segments_path: Path to preprocessed segments pickle file
            prompt_bars: Number of bars for prompt
            continuation_bars: Number of bars to continue
            seq_len: Maximum sequence length
        """
        self.prompt_bars = prompt_bars
        self.continuation_bars = continuation_bars
        self.seq_len = seq_len

        # Load segments
        with open(segments_path, 'rb') as f:
            self.segments = pickle.load(f)

        print(f"Loaded {len(self.segments)} segments for conditional generation")

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        """Get a conditional training sample."""
        tokens = self.segments[idx]

        # This is a simplified version
        # Ideally, split by bar markers
        mid_point = len(tokens) // 3  # Roughly 1/3 for prompt

        prompt = tokens[:mid_point]
        continuation = tokens[mid_point:mid_point + self.seq_len]

        prompt_tensor = torch.tensor(prompt, dtype=torch.long)
        continuation_tensor = torch.tensor(continuation, dtype=torch.long)

        return prompt_tensor, continuation_tensor


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate function for DataLoader.
    Handles variable-length sequences by padding.
    """
    inputs, targets = zip(*batch)

    # Find max length in batch
    max_len = max(inp.size(0) for inp in inputs)

    # Pad sequences
    padded_inputs = []
    padded_targets = []

    for inp, tgt in zip(inputs, targets):
        # Pad with 0 (assuming 0 is PAD token)
        pad_len = max_len - inp.size(0)
        if pad_len > 0:
            inp = torch.cat([inp, torch.zeros(pad_len, dtype=torch.long)])
            tgt = torch.cat([tgt, torch.zeros(pad_len, dtype=torch.long)])

        padded_inputs.append(inp)
        padded_targets.append(tgt)

    # Stack into batch
    inputs_batch = torch.stack(padded_inputs)
    targets_batch = torch.stack(padded_targets)

    return inputs_batch, targets_batch


def get_dataloaders(
    segments_path: str,
    batch_size: int = 16,
    seq_len: int = 512,
    num_workers: int = 4,
    train_split: float = 0.95,
    seed: int = 42
):
    """
    Create train and validation dataloaders.

    Args:
        segments_path: Path to preprocessed segments
        batch_size: Batch size
        seq_len: Sequence length
        num_workers: Number of dataloader workers
        train_split: Fraction of data for training
        seed: Random seed for splitting

    Returns:
        train_loader, val_loader
    """
    from torch.utils.data import DataLoader, random_split

    # Create full dataset
    dataset = MusicDataset(
        segments_path=segments_path,
        seq_len=seq_len,
        augment=True
    )

    # Split into train and validation
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=generator
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    return train_loader, val_loader
