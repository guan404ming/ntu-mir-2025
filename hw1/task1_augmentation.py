import torch
import torchaudio
import torchaudio.transforms as T
import json
import os
from tqdm import tqdm
import warnings
import pickle
from datetime import datetime

warnings.filterwarnings("ignore")


class AudioAugmentationPipeline:
    """
    Advanced audio augmentation pipeline for music data
    """

    def __init__(self, sr=16000, device=None, checkpoint_dir="checkpoints"):
        self.sr = sr
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.checkpoint_dir = checkpoint_dir

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        print(f"Audio Augmentation Pipeline initialized on {self.device}")
        print(f"Checkpoint directory: {self.checkpoint_dir}")

        # Traditional augmentation transforms
        self._init_traditional_transforms()

    def _init_traditional_transforms(self):
        """Initialize traditional audio augmentation transforms"""
        self.traditional_transforms = {
            "pitch_shift": T.PitchShift(self.sr, n_steps=2).to(self.device),
            "time_stretch": T.TimeStretch(hop_length=512, n_freq=1024).to(self.device),
            "add_noise": self._add_noise,
            "speed_change": self._speed_change,
        }

    def apply_traditional_augmentations(self, waveform):
        """Apply traditional audio augmentations with memory management"""
        augmented_versions = []

        try:
            # Clear GPU cache before processing
            if self.device == "cuda":
                torch.cuda.empty_cache()

            # Move waveform to device once
            waveform = waveform.to(self.device)

            with torch.no_grad():  # Disable gradients to save memory
                # Pitch shifting (reduced to save memory)
                for n_steps in [-1, 1]:  # Reduced from [-2, -1, 1, 2]
                    try:
                        pitch_shifter = T.PitchShift(self.sr, n_steps=n_steps).to(
                            self.device
                        )
                        shifted = pitch_shifter(waveform)
                        augmented_versions.append(
                            shifted.cpu()
                        )  # Move to CPU immediately
                        del shifted, pitch_shifter  # Explicit cleanup
                        if self.device == "cuda":
                            torch.cuda.empty_cache()
                    except RuntimeError as e:
                        if "CUDA out of memory" in str(e):
                            print(f"Skipping pitch shift {n_steps} due to memory")
                            continue
                        raise

                # Add noise (reduced levels)
                for noise_level in [0.01]:  # Reduced from [0.005, 0.01, 0.02]
                    try:
                        noisy = self._add_noise(waveform, noise_level)
                        augmented_versions.append(noisy.cpu())
                        del noisy
                        if self.device == "cuda":
                            torch.cuda.empty_cache()
                    except RuntimeError as e:
                        if "CUDA out of memory" in str(e):
                            print("Skipping noise augmentation due to memory")
                            continue
                        raise

                # Speed changes (reduced options)
                for speed_factor in [0.95, 1.05]:  # Reduced from [0.9, 0.95, 1.05, 1.1]
                    try:
                        speed_changed = self._speed_change(waveform, speed_factor)
                        augmented_versions.append(speed_changed.cpu())
                        del speed_changed
                        if self.device == "cuda":
                            torch.cuda.empty_cache()
                    except RuntimeError as e:
                        if "CUDA out of memory" in str(e):
                            print(f"Skipping speed change {speed_factor} due to memory")
                            continue
                        raise

        except Exception as e:
            print(f"Traditional augmentation failed: {e}")
            # Clear cache on error
            if self.device == "cuda":
                torch.cuda.empty_cache()

        return augmented_versions

    def _add_noise(self, waveform, noise_level=0.01):
        """Add gaussian noise to waveform"""
        noise = torch.randn_like(waveform) * noise_level
        return waveform + noise

    def _speed_change(self, waveform, speed_factor):
        """Change speed of audio without changing pitch"""
        try:
            # Use torchaudio's speed transform
            effects = [["speed", str(speed_factor)], ["rate", str(self.sr)]]
            augmented, _ = torchaudio.sox_effects.apply_effects_tensor(
                waveform, self.sr, effects
            )
            return augmented
        except Exception:
            # Fallback: simple resampling
            original_length = waveform.shape[-1]
            new_length = int(original_length / speed_factor)
            return torch.nn.functional.interpolate(
                waveform.unsqueeze(0), size=new_length, mode="linear"
            ).squeeze(0)

    def augment_dataset(
        self,
        audio_paths,
        augmentation_factor=2,
        batch_size=1,
        resume_from_checkpoint=True,
        checkpoint_interval=10,
    ):
        """
        Augment an entire dataset with memory management and checkpoint support

        Args:
            audio_paths: List of audio file paths
            augmentation_factor: Number of augmented versions per original
            batch_size: Number of files to process before clearing memory
            resume_from_checkpoint: Whether to resume from existing checkpoint
            checkpoint_interval: Save checkpoint every N batches

        Returns:
            List of augmented data dictionaries
        """
        augmented_data = []
        start_batch = 0
        processed_files = 0

        # Try to load checkpoint if resuming
        if resume_from_checkpoint:
            checkpoint_data = self.load_checkpoint()
            if checkpoint_data:
                # Check if checkpoint has complete audio data
                if "augmented_data" in checkpoint_data:
                    # Checkpoint has complete data - use it
                    augmented_data = checkpoint_data["augmented_data"]
                    print(f"📂 Loaded {len(augmented_data)} samples from checkpoint")
                elif "augmented_metadata" in checkpoint_data:
                    # Old metadata-only checkpoint - try to load final checkpoint
                    processed_files = checkpoint_data.get("processed_files", 0)
                    if processed_files >= len(audio_paths):
                        print(
                            "✅ All files already processed! Loading final checkpoint..."
                        )
                        final_checkpoint = self.load_checkpoint(
                            checkpoint_name="augmentation_final"
                        )
                        if final_checkpoint and "augmented_data" in final_checkpoint:
                            augmented_data = final_checkpoint["augmented_data"]
                            print(
                                f"📂 Loaded {len(augmented_data)} samples from final checkpoint"
                            )
                            return augmented_data
                        else:
                            print(
                                "⚠️  Final checkpoint not found, will regenerate from scratch"
                            )
                            augmented_data = []
                    else:
                        print("⚠️  Old metadata-only checkpoint found. Will regenerate.")
                        augmented_data = []

                start_batch = checkpoint_data.get("current_batch", 0)
                processed_files = checkpoint_data.get("processed_files", 0)

                print("🔄 Resuming from checkpoint:")
                print(f"   - Processed files: {processed_files}/{len(audio_paths)}")
                print(f"   - Starting from batch: {start_batch}")
                print(f"   - Existing augmented data: {len(augmented_data)} samples")

        print(f"Augmenting {len(audio_paths)} audio files...")
        print(f"Using device: {self.device}")
        print(f"Batch size: {batch_size} files")
        print(f"Checkpoint interval: {checkpoint_interval} batches")

        with tqdm(
            total=len(audio_paths),
            desc="Augmenting dataset",
            unit="file",
            colour="blue",
            initial=processed_files,
        ) as pbar:
            for batch_idx in range(start_batch, len(audio_paths), batch_size):
                batch_start = batch_idx
                batch_end = min(batch_start + batch_size, len(audio_paths))
                batch_paths = audio_paths[batch_start:batch_end]

                # Clear GPU memory at start of each batch
                if self.device == "cuda":
                    torch.cuda.empty_cache()

                for i, audio_path in enumerate(batch_paths):
                    current_file_idx = batch_start + i

                    # Skip if already processed (when resuming)
                    if current_file_idx < processed_files:
                        continue

                    try:
                        # Load original audio
                        waveform, orig_sr = torchaudio.load(audio_path)

                        if orig_sr != self.sr:
                            resampler = T.Resample(orig_sr, self.sr)
                            waveform = resampler(waveform)

                        if waveform.shape[0] > 1:
                            waveform = torch.mean(waveform, dim=0, keepdim=True)

                        # Store original
                        augmented_data.append(
                            {
                                "waveform": waveform.clone(),
                                "label": self._extract_label_from_path(audio_path),
                                "augmentation_type": "original",
                                "source_file": audio_path,
                            }
                        )

                        # Generate augmentations
                        augmentations_created = 0

                        # Traditional augmentations
                        if augmentations_created < augmentation_factor:
                            traditional_versions = self.apply_traditional_augmentations(
                                waveform
                            )

                            for j, aug_waveform in enumerate(traditional_versions):
                                if augmentations_created < augmentation_factor:
                                    augmented_data.append(
                                        {
                                            "waveform": aug_waveform.clone(),
                                            "label": self._extract_label_from_path(
                                                audio_path
                                            ),
                                            "augmentation_type": f"traditional_{j}",
                                            "source_file": audio_path,
                                        }
                                    )
                                    augmentations_created += 1

                        processed_files += 1

                        pbar.set_postfix(
                            {
                                "augmented": f"{augmentations_created}",
                                "total": f"{len(augmented_data)}",
                                "batch": f"{batch_start // batch_size + 1}",
                            }
                        )

                        # Clear variables
                        del waveform, traditional_versions

                    except Exception as e:
                        print(f"Failed to augment {audio_path}: {e}")

                    pbar.update(1)

                # Save checkpoint periodically
                current_batch_num = batch_start // batch_size + 1
                if current_batch_num % checkpoint_interval == 0:
                    checkpoint_data = self.create_checkpoint_data(
                        processed_files,
                        len(audio_paths),
                        augmented_data,
                        batch_end,
                        augmentation_factor,
                    )
                    self.save_checkpoint(checkpoint_data)

                # Force garbage collection and clear GPU memory after each batch
                if self.device == "cuda":
                    torch.cuda.empty_cache()

        # Save final checkpoint with complete data
        final_checkpoint_data = self.create_final_checkpoint_data(
            processed_files,
            len(audio_paths),
            augmented_data,
            len(audio_paths),
            augmentation_factor,
        )
        self.save_checkpoint(final_checkpoint_data, "augmentation_final")

        print(f"Dataset augmentation complete. Created {len(augmented_data)} samples.")
        return augmented_data

    def augment_dataset_streaming(
        self,
        audio_paths,
        split_name,
        output_dir,
        augmentation_factor=3,
        resume_from_checkpoint=True,
    ):
        """
        Stream augmentation: process one file at a time and save immediately to disk
        Use existing files on disk as checkpoint system

        Args:
            audio_paths: List of audio file paths to process
            split_name: 'train' or 'val'
            output_dir: Base output directory
            augmentation_factor: Number of augmented versions per original
            resume_from_checkpoint: Whether to check existing files and resume

        Returns:
            List of manifest entries for saved files
        """
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        manifest = []
        processed_files = 0

        # Check existing files if resuming
        if resume_from_checkpoint:
            existing_files = [f for f in os.listdir(split_dir) if f.endswith(".wav")]
            processed_files = len(existing_files) // (
                augmentation_factor + 1
            )  # +1 for original
            print(
                f"🔄 Found {len(existing_files)} existing files, resuming from file {processed_files}"
            )

        print(f"🎵 Streaming augmentation for {split_name}: {len(audio_paths)} files")
        print(f"📁 Output directory: {split_dir}")
        print(f"🔄 Augmentation factor: {augmentation_factor}")

        with tqdm(
            total=len(audio_paths),
            desc=f"Augmenting {split_name}",
            unit="file",
            colour="blue",
            initial=processed_files,
        ) as pbar:
            for i, audio_path in enumerate(audio_paths):
                # Skip if already processed
                if i < processed_files:
                    continue

                try:
                    # Load and preprocess audio
                    waveform, orig_sr = torchaudio.load(audio_path)
                    if orig_sr != self.sr:
                        resampler = T.Resample(orig_sr, self.sr)
                        waveform = resampler(waveform)
                    if waveform.shape[0] > 1:
                        waveform = torch.mean(waveform, dim=0, keepdim=True)

                    label = self._extract_label_from_path(audio_path)

                    # Save original
                    orig_filename = f"{split_name}_{i:06d}_{label}_original.wav"
                    orig_filepath = os.path.join(split_dir, orig_filename)
                    torchaudio.save(orig_filepath, waveform, self.sr)

                    manifest.append(
                        {
                            "file": os.path.join(split_name, orig_filename),
                            "label": label,
                            "augmentation_type": "original",
                            "original_index": i,
                            "source_file": audio_path,
                        }
                    )

                    # Generate and save augmentations
                    augmented_versions = self.apply_traditional_augmentations(waveform)

                    for j, aug_waveform in enumerate(augmented_versions):
                        if j >= augmentation_factor:
                            break

                        aug_filename = f"{split_name}_{i:06d}_{label}_aug_{j}.wav"
                        aug_filepath = os.path.join(split_dir, aug_filename)
                        torchaudio.save(aug_filepath, aug_waveform, self.sr)

                        manifest.append(
                            {
                                "file": os.path.join(split_name, aug_filename),
                                "label": label,
                                "augmentation_type": f"traditional_{j}",
                                "original_index": i,
                                "source_file": audio_path,
                            }
                        )

                    # Clear memory
                    del waveform, augmented_versions
                    if self.device == "cuda":
                        torch.cuda.empty_cache()

                    pbar.update(1)

                except Exception as e:
                    print(f"❌ Failed to process {audio_path}: {e}")
                    pbar.update(1)
                    continue

        # Save manifest
        manifest_path = os.path.join(output_dir, f"{split_name}_manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"✅ Saved {len(manifest)} {split_name} files to {split_dir}")
        print(f"📄 Manifest saved to {manifest_path}")

        return manifest

    def _extract_label_from_path(self, file_path):
        """Extract artist name from file path"""
        parts = file_path.replace("\\", "/").split("/")
        try:
            train_val_idx = parts.index("train_val")
            return parts[train_val_idx + 1]
        except (ValueError, IndexError):
            return parts[-3] if len(parts) > 3 else "unknown"

    def save_checkpoint(
        self, checkpoint_data, checkpoint_name="augmentation_checkpoint"
    ):
        """
        Save checkpoint data to disk

        Args:
            checkpoint_data: Dictionary containing checkpoint information
            checkpoint_name: Name of the checkpoint file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = os.path.join(
            self.checkpoint_dir, f"{checkpoint_name}_{timestamp}.pkl"
        )

        try:
            with open(checkpoint_path, "wb") as f:
                pickle.dump(checkpoint_data, f)
            print(f"💾 Checkpoint saved: {checkpoint_path}")
            return checkpoint_path
        except Exception as e:
            print(f"❌ Failed to save checkpoint: {e}")
            return None

    def load_checkpoint(
        self, checkpoint_path=None, checkpoint_name="augmentation_checkpoint"
    ):
        """
        Load checkpoint data from disk

        Args:
            checkpoint_path: Specific checkpoint file path
            checkpoint_name: Name pattern to search for latest checkpoint

        Returns:
            Checkpoint data dictionary or None if not found
        """
        if checkpoint_path is None:
            checkpoint_files = [
                f
                for f in os.listdir(self.checkpoint_dir)
                if f.startswith(checkpoint_name) and f.endswith(".pkl")
            ]

            if not checkpoint_files:
                print(f"📂 No checkpoint files found with pattern: {checkpoint_name}")
                return None

            checkpoint_files.sort(reverse=True)
            checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_files[0])

        try:
            with open(checkpoint_path, "rb") as f:
                checkpoint_data = pickle.load(f)
            print(f"📂 Checkpoint loaded: {checkpoint_path}")
            return checkpoint_data
        except Exception as e:
            print(f"❌ Failed to load checkpoint: {e}")
            return None

    def create_checkpoint_data(
        self,
        processed_files,
        total_files,
        augmented_data,
        current_batch,
        augmentation_factor,
    ):
        """
        Create checkpoint data structure (with complete audio data)

        Args:
            processed_files: Number of files processed so far
            total_files: Total number of files to process
            augmented_data: Current augmented data list (with complete audio data)
            current_batch: Current batch index
            augmentation_factor: Augmentation factor used

        Returns:
            Checkpoint data dictionary with complete raw audio data
        """
        return {
            "processed_files": processed_files,
            "total_files": total_files,
            "augmented_data": augmented_data,  # Store complete data including waveforms
            "current_batch": current_batch,
            "augmentation_factor": augmentation_factor,
            "timestamp": datetime.now().isoformat(),
            "device": self.device,
            "sr": self.sr,
        }

    def create_final_checkpoint_data(
        self,
        processed_files,
        total_files,
        augmented_data,
        current_batch,
        augmentation_factor,
    ):
        """
        Create final checkpoint data structure (with complete audio data for loading)

        Args:
            processed_files: Number of files processed so far
            total_files: Total number of files to process
            augmented_data: Complete augmented data list (with actual audio)
            current_batch: Current batch index
            augmentation_factor: Augmentation factor used

        Returns:
            Final checkpoint data dictionary with complete data
        """
        return {
            "processed_files": processed_files,
            "total_files": total_files,
            "augmented_data": augmented_data,  # Complete data for final checkpoint
            "current_batch": current_batch,
            "augmentation_factor": augmentation_factor,
            "timestamp": datetime.now().isoformat(),
            "device": self.device,
            "sr": self.sr,
        }


def save_augmented_audio_to_disk(
    augmented_data, split="train", output_dir="data/artist20_augmented"
):
    """
    Save augmented audio data to disk as audio files

    Args:
        augmented_data: List of dicts with 'waveform', 'label', 'augmentation_type'
        split: Dataset split name ('train' or 'val')
        output_dir: Output directory for augmented files

    Returns:
        List of manifest entries
    """
    print(f"Saving {len(augmented_data)} augmented {split} files to disk...")

    split_dir = os.path.join(output_dir, split)
    os.makedirs(split_dir, exist_ok=True)

    # Create manifest file
    manifest = []

    for i, sample in enumerate(tqdm(augmented_data, desc=f"Saving {split} audio")):
        try:
            waveform = sample["waveform"]
            label = sample["label"]
            aug_type = sample["augmentation_type"]
            source_file = sample.get("source_file", "unknown")

            # Create filename
            filename = f"{split}_{i:06d}_{label}_{aug_type}.wav"
            filepath = os.path.join(split_dir, filename)

            # Save audio file
            torchaudio.save(filepath, waveform, 16000)

            # Add to manifest
            manifest.append(
                {
                    "file": os.path.join(split, filename),
                    "label": label,
                    "augmentation_type": aug_type,
                    "original_index": i,
                    "source_file": source_file,
                }
            )

        except Exception as e:
            print(f"Failed to save sample {i}: {e}")
            continue

    # Save manifest
    manifest_path = os.path.join(output_dir, f"{split}_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"✅ Saved {len(manifest)} {split} files to {split_dir}")
    print(f"📄 Manifest saved to {manifest_path}")

    return manifest


def create_augmented_dataset(
    train_files,
    val_files,
    base_dir="data/artist20/",
    augmentation_factor=3,
    use_gpu=True,
    output_dir="data/artist20_augmented/",
    resume_from_checkpoint=True,
):
    """
    Create augmented dataset with traditional augmentations and checkpoint support

    Args:
        train_files: List of training file paths
        val_files: List of validation file paths
        base_dir: Base directory for audio files
        augmentation_factor: Number of augmented versions per original
        use_gpu: Whether to use GPU acceleration
        output_dir: Directory to save augmented audio files
        resume_from_checkpoint: Whether to resume from existing checkpoint

    Returns:
        Tuple of (augmented_train_data, augmented_val_data, train_manifest, val_manifest)
    """
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"

    # Checkpoint directories will be created for each pipeline separately

    # Prepare full paths
    train_paths = []
    for file_path in train_files:
        clean_path = file_path.lstrip("./")
        full_path = os.path.join(base_dir, clean_path)
        if os.path.exists(full_path):
            train_paths.append(full_path)

    val_paths = []
    for file_path in val_files:
        clean_path = file_path.lstrip("./")
        full_path = os.path.join(base_dir, clean_path)
        if os.path.exists(full_path):
            val_paths.append(full_path)

    # Use streaming augmentation: process one file at a time and save immediately
    pipeline = AudioAugmentationPipeline(device=device)

    print(f"🎵 Streaming augmentation for training set ({len(train_paths)} files)...")
    train_manifest = pipeline.augment_dataset_streaming(
        train_paths,
        "train",
        output_dir,
        augmentation_factor=augmentation_factor,
        resume_from_checkpoint=resume_from_checkpoint,
    )

    print(f"🎵 Streaming augmentation for validation set ({len(val_paths)} files)...")
    val_manifest = pipeline.augment_dataset_streaming(
        val_paths,
        "val",
        output_dir,
        augmentation_factor=augmentation_factor,
        resume_from_checkpoint=resume_from_checkpoint,
    )

    # Create dummy data for compatibility with the rest of the function
    augmented_train_data = (
        [{"dummy": "data"}] * len(train_manifest) if train_manifest else []
    )
    augmented_val_data = [{"dummy": "data"}] * len(val_manifest) if val_manifest else []

    # Save dataset summary
    summary = {
        "original_train_files": len(train_files),
        "original_val_files": len(val_files),
        "augmented_train_samples": len(augmented_train_data),
        "augmented_val_samples": len(augmented_val_data),
        "augmentation_factor": augmentation_factor,
        "output_directory": output_dir,
    }

    summary_path = os.path.join(output_dir, "dataset_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"📊 Dataset summary saved to {summary_path}")

    return augmented_train_data, augmented_val_data, train_manifest, val_manifest


def main():
    """Main augmentation pipeline"""
    print("🎵 Audio Data Augmentation Pipeline")
    print("=" * 50)

    # Load training and validation data
    try:
        with open("data/artist20/train.json", "r") as f:
            train_files = json.load(f)

        with open("data/artist20/val.json", "r") as f:
            val_files = json.load(f)

        print(f"Training files: {len(train_files)}")
        print(f"Validation files: {len(val_files)}")

    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please make sure you have the dataset files in data/artist20/")
        return

    # Get augmentation parameters (with defaults for non-interactive use)
    try:
        import sys

        if sys.stdin.isatty():
            # Interactive mode
            augmentation_factor = int(
                input("Enter augmentation factor (default=3): ") or "3"
            )
            save_to_disk_input = input(
                "Save augmented audio files to disk? (y/n, default=y): "
            ).lower()
            save_to_disk = (
                save_to_disk_input.startswith("y") if save_to_disk_input else True
            )

            output_dir = (
                input("Output directory (default=data/artist20_augmented/): ")
                or "data/artist20_augmented/"
            )

            resume_checkpoint_input = input(
                "Resume from existing files if available? (y/n, default=y): "
            ).lower()
            resume_from_checkpoint = (
                resume_checkpoint_input.startswith("y")
                if resume_checkpoint_input
                else True
            )
        else:
            # Non-interactive mode - use defaults
            print("Running in non-interactive mode with default parameters...")
            augmentation_factor = 3
            save_to_disk = True
            output_dir = "data/artist20_augmented/"
            resume_from_checkpoint = True  # Enable resume from existing files

    except (KeyboardInterrupt, EOFError):
        print("\n❌ Augmentation cancelled by user")
        return

    print("\nConfiguration:")
    print(f"  Augmentation factor: {augmentation_factor}")
    print(f"  Save to disk: {save_to_disk}")
    print(f"  Output directory: {output_dir}")
    print(f"  Resume from existing files: {resume_from_checkpoint}")
    print()

    try:
        # Create augmented dataset
        (
            augmented_train_data,
            augmented_val_data,
            train_manifest,
            val_manifest,
        ) = create_augmented_dataset(
            train_files,
            val_files,
            augmentation_factor=augmentation_factor,
            save_to_disk=save_to_disk,
            output_dir=output_dir,
            resume_from_checkpoint=resume_from_checkpoint,
        )

        print("\n✅ Audio Augmentation Completed Successfully!")
        print("=" * 50)
        print(f"📁 Original dataset: {len(train_files)} train + {len(val_files)} val")
        print(
            f"🔄 Augmented dataset: {len(augmented_train_data)} train + {len(augmented_val_data)} val"
        )
        if len(train_files) > 0:
            print(
                f"📈 Augmentation factor: {len(augmented_train_data) / len(train_files):.1f}x for training, "
                f"{len(augmented_val_data) / len(val_files):.1f}x for validation"
            )
        else:
            print(
                f"📈 Augmentation factor: {len(augmented_val_data) / len(val_files):.1f}x for validation"
            )

        if save_to_disk:
            print(f"🎵 Augmented audio files saved to: {output_dir}")
            print("📄 Manifest files:")
            print(f"  - {output_dir}/train_manifest.json")
            print(f"  - {output_dir}/val_manifest.json")
            print(f"  - {output_dir}/dataset_summary.json")

        print("\n🔄 Next steps:")
        print("  1. Run task1_preprocessing_augmented.py to extract features")
        print("  2. Run task1_train_augmented.py to train models")

    except Exception as e:
        print(f"❌ Error during augmentation: {e}")
        import traceback

        traceback.print_exc()


def test_checkpoint_functionality():
    """Test checkpoint save and load functionality"""
    print("🧪 Testing checkpoint functionality...")

    # Create test pipeline
    test_pipeline = AudioAugmentationPipeline(checkpoint_dir="test_checkpoints")

    # Create test checkpoint data
    test_data = {
        "processed_files": 50,
        "total_files": 100,
        "augmented_data": [{"test": "data"}],
        "current_batch": 16,
        "augmentation_factor": 3,
    }

    # Test save
    checkpoint_path = test_pipeline.save_checkpoint(test_data, "test_checkpoint")
    if checkpoint_path:
        print("✅ Checkpoint save test passed")
    else:
        print("❌ Checkpoint save test failed")
        return False

    # Test load
    loaded_data = test_pipeline.load_checkpoint(checkpoint_path)
    if loaded_data and loaded_data["processed_files"] == 50:
        print("✅ Checkpoint load test passed")
    else:
        print("❌ Checkpoint load test failed")
        return False

    # Cleanup test files
    try:
        import shutil

        shutil.rmtree("test_checkpoints")
        print("✅ Test cleanup completed")
    except Exception as e:
        print(f"⚠️  Test cleanup warning: {e}")

    print("🎉 All checkpoint tests passed!")
    return True


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_checkpoint_functionality()
    else:
        main()
