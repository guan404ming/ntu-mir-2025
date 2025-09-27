from task1_augmentation import AudioAugmentationPipeline
import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
import json
import os
from tqdm import tqdm
import warnings
import time
from datetime import timedelta

warnings.filterwarnings("ignore")


class AugmentedAudioFeatureExtractor:
    """
    Enhanced audio feature extractor with augmentation support
    """

    def __init__(self, sr=16000, use_augmentation=True, augmentation_factor=3):
        self.sr = sr
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_augmentation = use_augmentation
        self.augmentation_factor = augmentation_factor

        # GPU memory optimization
        if self.device == "cuda":
            self.gpu_memory = torch.cuda.get_device_properties(0).total_memory
            self.optimal_batch_size = self._calculate_optimal_batch_size()
            print(
                f"GPU Memory: {self.gpu_memory / 1e9:.1f}GB, Optimal batch size: {self.optimal_batch_size}"
            )
        else:
            self.optimal_batch_size = 4

        print(f"Augmented Feature Extractor initialized on {self.device}")
        print(f"Augmentation: {'Enabled' if use_augmentation else 'Disabled'}")

        # Initialize feature extraction transforms
        self.mfcc_transform = T.MFCC(
            sample_rate=sr,
            n_mfcc=20,
            melkwargs={"n_fft": 2048, "hop_length": 512, "n_mels": 128},
        ).to(self.device)

        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sr, n_fft=2048, hop_length=512, n_mels=128
        ).to(self.device)

        self.spectral_centroid = T.SpectralCentroid(
            sample_rate=sr, n_fft=2048, hop_length=512
        ).to(self.device)

        # Initialize augmentation pipeline if requested
        if self.use_augmentation:
            self.augmentation_pipeline = AudioAugmentationPipeline(
                sr=sr, device=self.device
            )

    def _calculate_optimal_batch_size(self):
        """Calculate optimal batch size based on GPU memory"""
        if self.device == "cpu":
            return 4

        # Rough estimation: 1GB GPU mem can handle ~8 samples
        gb_memory = self.gpu_memory / 1e9
        if gb_memory >= 8:
            return 16
        elif gb_memory >= 4:
            return 12
        elif gb_memory >= 2:
            return 8
        else:
            return 4

    def extract_features_from_augmented_data(
        self,
        augmented_data,
        batch_size=None,
        checkpoint_dir="results/task1_augmented/checkpoints",
        lazy_loading=True,
    ):
        """
        Extract features from pre-augmented data with checkpoint support

        Args:
            augmented_data: List of dicts with 'waveform'/'file_path', 'label', 'augmentation_type'
            batch_size: Batch size for processing (auto-detected if None)
            checkpoint_dir: Directory to save checkpoints
            lazy_loading: Whether to use lazy loading for memory efficiency

        Returns:
            Features and labels arrays
        """
        if batch_size is None:
            batch_size = self.optimal_batch_size

        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(
            checkpoint_dir, "augmented_features_checkpoint.json"
        )

        # Load existing checkpoint if available
        start_idx = 0
        all_features = []
        all_labels = []

        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, "r") as f:
                checkpoint = json.load(f)
                start_idx = checkpoint.get("processed_samples", 0)

            # Load existing features if checkpoint exists and has processed samples
            if start_idx > 0:
                features_path = os.path.join(checkpoint_dir, "features_partial.npy")
                labels_path = os.path.join(checkpoint_dir, "labels_partial.npy")

                if os.path.exists(features_path) and os.path.exists(labels_path):
                    all_features = np.load(features_path).tolist()
                    all_labels = np.load(labels_path).tolist()
                    print(
                        f"📁 Resuming from checkpoint: {start_idx}/{len(augmented_data)} samples processed"
                    )

        print(f"Extracting features from {len(augmented_data)} augmented samples...")
        print(f"Starting from sample {start_idx}")
        print(f"Batch size: {batch_size} (GPU optimized: {self.device})")

        start_time = time.time()
        with tqdm(
            initial=start_idx,
            total=len(augmented_data),
            desc="Feature extraction",
            unit="sample",
            colour="green",
        ) as pbar:
            for i in range(start_idx, len(augmented_data), batch_size):
                if lazy_loading and "file_path" in augmented_data[0]:
                    # Lazy load batch of audio data
                    batch_data = load_augmented_batch_lazy(
                        augmented_data, i, batch_size
                    )
                else:
                    # Use pre-loaded data
                    batch_data = augmented_data[i : i + batch_size]

                batch_features, batch_labels = self._process_augmented_batch(batch_data)
                all_features.extend(batch_features)
                all_labels.extend(batch_labels)
                pbar.update(len(batch_data))

                # Calculate ETA and throughput
                elapsed = time.time() - start_time
                current_sample = i + len(batch_data)
                remaining_samples = len(augmented_data) - current_sample
                if current_sample > start_idx:
                    samples_per_sec = (current_sample - start_idx) / elapsed
                    eta_seconds = (
                        remaining_samples / samples_per_sec
                        if samples_per_sec > 0
                        else 0
                    )
                    eta_str = str(timedelta(seconds=int(eta_seconds)))
                else:
                    samples_per_sec = 0
                    eta_str = "calculating..."

                pbar.set_postfix(
                    {
                        "extracted": len(all_features),
                        "failed": len(batch_data) - len(batch_features),
                        "rate": f"{samples_per_sec:.1f}/s",
                        "ETA": eta_str,
                    }
                )

                # Save checkpoint every 10 batches
                if (i - start_idx) % (batch_size * 10) == 0:
                    self._save_checkpoint(
                        checkpoint_path,
                        checkpoint_dir,
                        i + len(batch_data),
                        all_features,
                        all_labels,
                    )

        # Clean up checkpoint files
        self._cleanup_checkpoint(checkpoint_path, checkpoint_dir)

        return np.array(all_features), np.array(all_labels)

    def _process_augmented_batch(self, batch_data):
        """Process a batch of augmented audio data with memory management"""
        batch_features = []
        batch_labels = []

        # Clear GPU cache before batch processing
        if self.device == "cuda":
            torch.cuda.empty_cache()

        with torch.no_grad():
            for sample in batch_data:
                try:
                    waveform = sample["waveform"].to(self.device)
                    label = sample["label"]

                    # Extract features from waveform
                    features = self._extract_single_features(waveform)

                    if features is not None:
                        # features is already a list, not a tensor
                        batch_features.append(features)
                        batch_labels.append(label)

                    # Clear waveform from GPU memory
                    del waveform
                    if self.device == "cuda":
                        torch.cuda.empty_cache()

                except Exception as e:
                    print(f"Feature extraction failed for sample: {e}")
                    continue

        # Final cache clear after batch
        if self.device == "cuda":
            torch.cuda.empty_cache()

        return batch_features, batch_labels

    def _save_checkpoint(
        self, checkpoint_path, checkpoint_dir, processed_samples, features, labels
    ):
        """Save checkpoint for resumable processing"""
        try:
            # Save checkpoint metadata
            checkpoint_data = {
                "processed_samples": processed_samples,
                "total_features": len(features),
                "timestamp": str(
                    torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
                ),
            }
            with open(checkpoint_path, "w") as f:
                json.dump(checkpoint_data, f)

            # Save partial features and labels
            if features and labels:
                np.save(
                    os.path.join(checkpoint_dir, "features_partial.npy"),
                    np.array(features),
                )
                np.save(
                    os.path.join(checkpoint_dir, "labels_partial.npy"), np.array(labels)
                )
        except Exception as e:
            print(f"Warning: Failed to save checkpoint: {e}")

    def _cleanup_checkpoint(self, checkpoint_path, checkpoint_dir):
        """Clean up checkpoint files after successful completion"""
        try:
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
            for partial_file in ["features_partial.npy", "labels_partial.npy"]:
                partial_path = os.path.join(checkpoint_dir, partial_file)
                if os.path.exists(partial_path):
                    os.remove(partial_path)
        except Exception as e:
            print(f"Warning: Failed to cleanup checkpoint: {e}")

    def extract_features_with_augmentation(
        self,
        audio_paths,
        batch_size=None,
        checkpoint_dir="results/task1_augmented/checkpoints",
    ):
        """
        Extract features from audio files with on-the-fly augmentation with checkpoint support

        Args:
            audio_paths: List of audio file paths
            batch_size: Batch size for processing (auto-detected if None)
            checkpoint_dir: Directory to save checkpoints

        Returns:
            Features and labels arrays
        """
        if batch_size is None:
            batch_size = self.optimal_batch_size

        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, "on_the_fly_checkpoint.json")

        # Load existing checkpoint if available
        start_idx = 0
        all_features = []
        all_labels = []

        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, "r") as f:
                checkpoint = json.load(f)
                start_idx = checkpoint.get("processed_files", 0)

            # Load existing features if checkpoint exists and has processed files
            if start_idx > 0:
                features_path = os.path.join(checkpoint_dir, "features_otf_partial.npy")
                labels_path = os.path.join(checkpoint_dir, "labels_otf_partial.npy")

                if os.path.exists(features_path) and os.path.exists(labels_path):
                    all_features = np.load(features_path).tolist()
                    all_labels = np.load(labels_path).tolist()
                    print(
                        f"📁 Resuming from checkpoint: {start_idx}/{len(audio_paths)} files processed"
                    )

        print(f"Processing {len(audio_paths)} files with augmentation...")
        print(f"Starting from file {start_idx}")
        print(f"Batch size: {batch_size} (GPU optimized: {self.device})")

        start_time = time.time()
        with tqdm(
            initial=start_idx,
            total=len(audio_paths),
            desc="Augmented feature extraction",
            unit="file",
            colour="blue",
        ) as pbar:
            for i in range(start_idx, len(audio_paths), batch_size):
                batch_paths = audio_paths[i : i + batch_size]
                batch_features, batch_labels = self._process_batch_with_augmentation(
                    batch_paths
                )
                all_features.extend(batch_features)
                all_labels.extend(batch_labels)
                pbar.update(len(batch_paths))

                # Calculate ETA and throughput
                elapsed = time.time() - start_time
                current_file = i + len(batch_paths)
                remaining_files = len(audio_paths) - current_file
                if current_file > start_idx:
                    files_per_sec = (current_file - start_idx) / elapsed
                    eta_seconds = (
                        remaining_files / files_per_sec if files_per_sec > 0 else 0
                    )
                    eta_str = str(timedelta(seconds=int(eta_seconds)))
                else:
                    files_per_sec = 0
                    eta_str = "calculating..."

                pbar.set_postfix(
                    {
                        "samples": len(all_features),
                        "per_file": len(all_features) // max(1, current_file),
                        "rate": f"{files_per_sec:.2f}/s",
                        "ETA": eta_str,
                    }
                )

                # Save checkpoint every 5 batches for file processing
                if (i - start_idx) % (batch_size * 5) == 0:
                    self._save_otf_checkpoint(
                        checkpoint_path,
                        checkpoint_dir,
                        i + len(batch_paths),
                        all_features,
                        all_labels,
                    )

        # Clean up checkpoint files
        self._cleanup_otf_checkpoint(checkpoint_path, checkpoint_dir)

        return np.array(all_features), np.array(all_labels)

    def _process_batch_with_augmentation(self, batch_paths):
        """Process batch with on-the-fly augmentation and memory management"""
        batch_features = []
        batch_labels = []

        # Clear GPU cache before batch processing
        if self.device == "cuda":
            torch.cuda.empty_cache()

        with torch.no_grad():
            for audio_path in batch_paths:
                try:
                    # Load original audio
                    waveform, orig_sr = torchaudio.load(audio_path)

                    if orig_sr != self.sr:
                        resampler = T.Resample(orig_sr, self.sr)
                        waveform = resampler(waveform)

                    if waveform.shape[0] > 1:
                        waveform = torch.mean(waveform, dim=0, keepdim=True)

                    waveform = waveform.to(self.device)
                    label = self._extract_label_from_path(audio_path)

                    # Extract features from original
                    original_features = self._extract_single_features(waveform)
                    if original_features is not None:
                        batch_features.append(original_features)
                        batch_labels.append(label)

                    # Apply augmentations if enabled
                    if self.use_augmentation:
                        # Traditional augmentations (faster than source separation)
                        aug_waveforms = (
                            self.augmentation_pipeline.apply_traditional_augmentations(
                                waveform
                            )
                        )

                        # Limit augmentations to avoid memory issues
                        for aug_waveform in aug_waveforms[: self.augmentation_factor]:
                            try:
                                aug_waveform_gpu = aug_waveform.to(self.device)
                                aug_features = self._extract_single_features(
                                    aug_waveform_gpu
                                )
                                if aug_features is not None:
                                    batch_features.append(aug_features)
                                    batch_labels.append(label)

                                # Clear augmented waveform from GPU
                                del aug_waveform_gpu
                                if self.device == "cuda":
                                    torch.cuda.empty_cache()
                            except Exception:
                                continue

                        # Clear augmentation data
                        del aug_waveforms

                    # Clear original waveform from GPU memory
                    del waveform
                    if self.device == "cuda":
                        torch.cuda.empty_cache()

                except Exception:
                    continue

        # Final cache clear after batch
        if self.device == "cuda":
            torch.cuda.empty_cache()

        return batch_features, batch_labels

    def _save_otf_checkpoint(
        self, checkpoint_path, checkpoint_dir, processed_files, features, labels
    ):
        """Save checkpoint for on-the-fly processing"""
        try:
            # Save checkpoint metadata
            checkpoint_data = {
                "processed_files": processed_files,
                "total_features": len(features),
                "timestamp": str(
                    torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
                ),
            }
            with open(checkpoint_path, "w") as f:
                json.dump(checkpoint_data, f)

            # Save partial features and labels
            if features and labels:
                np.save(
                    os.path.join(checkpoint_dir, "features_otf_partial.npy"),
                    np.array(features),
                )
                np.save(
                    os.path.join(checkpoint_dir, "labels_otf_partial.npy"),
                    np.array(labels),
                )
        except Exception as e:
            print(f"Warning: Failed to save OTF checkpoint: {e}")

    def _cleanup_otf_checkpoint(self, checkpoint_path, checkpoint_dir):
        """Clean up on-the-fly checkpoint files after successful completion"""
        try:
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
            for partial_file in ["features_otf_partial.npy", "labels_otf_partial.npy"]:
                partial_path = os.path.join(checkpoint_dir, partial_file)
                if os.path.exists(partial_path):
                    os.remove(partial_path)
        except Exception as e:
            print(f"Warning: Failed to cleanup OTF checkpoint: {e}")

    def _extract_single_features(self, waveform):
        """Extract enhanced features from a single waveform"""
        try:
            features = {}

            # Enhanced MFCC features with more coefficients
            mfccs = self.mfcc_transform(waveform)
            features.update(self._compute_statistics(mfccs[0], "mfcc"))

            # Additional MFCC pooling operations
            if mfccs[0].dim() == 2 and mfccs[0].shape[1] > 1:
                # Global pooling across entire MFCC matrix
                mfcc_flat = mfccs[0].flatten()
                features["mfcc_global_energy"] = torch.sum(torch.abs(mfcc_flat)).cpu().item()
                features["mfcc_spectral_centroid"] = torch.sum(torch.arange(len(mfcc_flat), device=self.device, dtype=torch.float32) * torch.abs(mfcc_flat)) / (torch.sum(torch.abs(mfcc_flat)) + 1e-8)
                features["mfcc_spectral_centroid"] = features["mfcc_spectral_centroid"].cpu().item() if torch.is_tensor(features["mfcc_spectral_centroid"]) else features["mfcc_spectral_centroid"]

            # Mel spectrogram features
            mel_spec = self.mel_spectrogram(waveform)
            features.update(self._compute_statistics(mel_spec[0], "mel"))

            # Additional mel spectrogram pooling
            if mel_spec[0].dim() == 2 and mel_spec[0].shape[1] > 1:
                # Temporal texture features
                mel_diff_time = torch.diff(mel_spec[0], dim=1)
                features["mel_temporal_flux"] = torch.mean(torch.abs(mel_diff_time)).cpu().item()

                # Frequency texture features
                mel_diff_freq = torch.diff(mel_spec[0], dim=0)
                features["mel_freq_flux"] = torch.mean(torch.abs(mel_diff_freq)).cpu().item()

            # Enhanced Chroma-like features (low frequency bins)
            chroma_like = mel_spec[0, :12, :]
            features.update(self._compute_statistics(chroma_like, "chroma"))

            # Spectral centroid
            centroid = self.spectral_centroid(waveform)
            features.update(self._compute_statistics(centroid[0], "centroid"))

            # RMS energy
            rms = torch.sqrt(torch.mean(waveform**2, dim=-1, keepdim=True))
            features.update(self._compute_statistics(rms[0], "rms"))

            # Zero crossing rate
            sign_changes = torch.abs(torch.diff(torch.sign(waveform), dim=-1))
            zcr = torch.mean(sign_changes, dim=-1, keepdim=True)
            features.update(self._compute_statistics(zcr[0], "zcr"))

            # Spectral rolloff approximation
            mel_energy = torch.sum(mel_spec[0], dim=0, keepdim=True)
            features.update(self._compute_statistics(mel_energy, "rolloff"))

            # Enhanced tempo estimation with pooling
            if waveform.shape[-1] > self.sr:
                diff_spec = torch.diff(mel_spec[0], dim=1)
                onset_strength = torch.mean(torch.clamp(diff_spec, min=0), dim=0)

                # Enhanced tempo features with pooling
                features["tempo_mean"] = torch.mean(onset_strength).cpu().item() * 100 + 60
                features["tempo_std"] = torch.std(onset_strength).cpu().item() * 50
                features["tempo_max"] = torch.max(onset_strength).cpu().item() * 150 + 60

                # Rhythm regularity with pooling
                if len(onset_strength) > 1:
                    autocorr = torch.nn.functional.conv1d(
                        onset_strength.unsqueeze(0).unsqueeze(0),
                        onset_strength.unsqueeze(0).unsqueeze(0).flip(-1),
                        padding=onset_strength.shape[0] - 1,
                    )[0, 0]

                    mid_point = len(autocorr) // 2
                    rhythm_peaks = autocorr[mid_point:]
                    features["rhythm_regularity_max"] = torch.max(rhythm_peaks).cpu().item()
                    features["rhythm_regularity_mean"] = torch.mean(rhythm_peaks).cpu().item()
                    features["rhythm_regularity_std"] = torch.std(rhythm_peaks).cpu().item()

                # Enhanced onset detection pooling
                onset_strength_smooth = torch.nn.functional.conv1d(
                    onset_strength.unsqueeze(0).unsqueeze(0),
                    torch.ones(1, 1, 3, device=self.device) / 3.0,
                    padding=1
                )[0, 0]
                features["onset_density"] = torch.sum(onset_strength > torch.mean(onset_strength)).cpu().item() / len(onset_strength)
                features["onset_regularity"] = torch.std(torch.diff(torch.nonzero(onset_strength > torch.mean(onset_strength)).flatten().float())).cpu().item() if torch.sum(onset_strength > torch.mean(onset_strength)) > 1 else 0.0

            # Enhanced spectral bandwidth with pooling
            freq_weighted = torch.arange(
                mel_spec.shape[1], dtype=torch.float32, device=self.device
            ).unsqueeze(1)
            bandwidth = torch.sum(freq_weighted * mel_spec[0], dim=0) / (
                torch.sum(mel_spec[0], dim=0) + 1e-8
            )
            features.update(
                self._compute_statistics(bandwidth.unsqueeze(0), "bandwidth")
            )

            # Additional spectral shape features with pooling
            if mel_spec[0].shape[1] > 1:
                # Spectral rolloff with multiple percentiles
                total_energy = torch.cumsum(mel_spec[0], dim=0)
                total_sum = torch.sum(mel_spec[0], dim=0, keepdim=True)

                rolloff_85 = torch.argmax((total_energy / (total_sum + 1e-8) > 0.85).float(), dim=0).float()
                rolloff_95 = torch.argmax((total_energy / (total_sum + 1e-8) > 0.95).float(), dim=0).float()

                features["rolloff_85_mean"] = torch.mean(rolloff_85).cpu().item()
                features["rolloff_85_std"] = torch.std(rolloff_85).cpu().item()
                features["rolloff_95_mean"] = torch.mean(rolloff_95).cpu().item()
                features["rolloff_95_std"] = torch.std(rolloff_95).cpu().item()

            # Delta MFCC features
            mfcc_delta = torch.diff(mfccs[0], dim=1, prepend=mfccs[0, :, :1])
            features.update(self._compute_statistics(mfcc_delta, "mfcc_delta"))

            # Delta-delta MFCC features
            mfcc_delta2 = torch.diff(mfcc_delta, dim=1, prepend=mfcc_delta[:, :1])
            features.update(self._compute_statistics(mfcc_delta2, "mfcc_delta2"))

            # Spectral contrast features
            contrast = self._compute_spectral_contrast(mel_spec[0])
            features.update(self._compute_statistics(contrast, "contrast"))

            # NEW: Enhanced spectral features
            # Spectral flatness
            geometric_mean = torch.exp(torch.mean(torch.log(mel_spec[0] + 1e-8), dim=0))
            arithmetic_mean = torch.mean(mel_spec[0], dim=0)
            flatness = geometric_mean / (arithmetic_mean + 1e-8)
            features.update(self._compute_statistics(flatness.unsqueeze(0), "flatness"))

            # NEW: Spectral skewness and kurtosis
            mel_centered = mel_spec[0] - torch.mean(mel_spec[0], dim=1, keepdim=True)
            mel_std = torch.std(mel_spec[0], dim=1, keepdim=True) + 1e-8
            mel_normalized = mel_centered / mel_std

            skewness = torch.mean(mel_normalized**3, dim=1)
            kurtosis = torch.mean(mel_normalized**4, dim=1) - 3
            features.update(self._compute_statistics(skewness.unsqueeze(0), "skewness"))
            features.update(self._compute_statistics(kurtosis.unsqueeze(0), "kurtosis"))

            # NEW: Harmonic-percussive separation approximation
            # Vertical (harmonic) vs horizontal (percussive) energy
            mel_harmonic = torch.mean(mel_spec[0], dim=1)  # Freq average
            mel_percussive = torch.mean(mel_spec[0], dim=0)  # Time average
            features.update(
                self._compute_statistics(mel_harmonic.unsqueeze(0), "harmonic")
            )
            features.update(
                self._compute_statistics(mel_percussive.unsqueeze(0), "percussive")
            )

            # NEW: Spectral energy distribution
            total_energy = torch.sum(mel_spec[0])
            if total_energy > 0:
                low_energy = torch.sum(mel_spec[0, :32, :]) / total_energy
                mid_energy = torch.sum(mel_spec[0, 32:96, :]) / total_energy
                high_energy = torch.sum(mel_spec[0, 96:, :]) / total_energy
                features["low_energy_ratio"] = low_energy.cpu().item()
                features["mid_energy_ratio"] = mid_energy.cpu().item()
                features["high_energy_ratio"] = high_energy.cpu().item()

            # NEW: Onset detection features
            if waveform.shape[-1] > self.sr:
                # Spectral flux
                spec_diff = torch.diff(mel_spec[0], dim=1)
                spectral_flux = torch.mean(torch.clamp(spec_diff, min=0), dim=0)
                features.update(
                    self._compute_statistics(spectral_flux.unsqueeze(0), "flux")
                )

                # Complex domain features
                torch.abs(mel_spec[0])
                spec_phase = torch.angle(
                    torch.complex(mel_spec[0], torch.zeros_like(mel_spec[0]))
                )
                phase_deviation = torch.diff(spec_phase, dim=1)
                features.update(self._compute_statistics(phase_deviation, "phase_dev"))

            # NEW: Timbral texture features
            # Roughness approximation
            roughness = torch.std(mel_spec[0], dim=1)
            features.update(
                self._compute_statistics(roughness.unsqueeze(0), "roughness")
            )

            # Brightness (spectral centroid normalized)
            if torch.sum(mel_spec[0]) > 0:
                brightness = torch.sum(
                    freq_weighted.squeeze() * torch.mean(mel_spec[0], dim=1)
                ) / torch.sum(torch.mean(mel_spec[0], dim=1))
                features["brightness"] = brightness.cpu().item()

            # Convert to feature array with enhanced stability
            feature_array = []
            for key in sorted(features.keys()):
                value = features[key]
                if isinstance(value, torch.Tensor):
                    val = value.cpu().item()
                else:
                    val = float(value)

                # Enhanced handling of NaN, inf values with robust replacement
                if np.isnan(val) or np.isinf(val):
                    val = 0.0

                # Additional stability: clip extreme values
                val = np.clip(val, -1e6, 1e6)

                feature_array.append(val)

            return feature_array

        except Exception:
            return None

    def _compute_statistics(self, tensor, prefix):
        """Compute enhanced statistical features with pooling operations"""
        stats = {}

        if tensor.dim() == 0:
            val = tensor.cpu().item()
            stats[f"{prefix}_mean"] = 0.0 if (np.isnan(val) or np.isinf(val)) else val
            return stats

        if tensor.dim() == 1 or (tensor.dim() == 2 and tensor.shape[0] == 1):
            flat = tensor.flatten()
            flat = torch.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)

            # Basic statistics
            stats[f"{prefix}_mean"] = torch.mean(flat).cpu().item()
            stats[f"{prefix}_std"] = torch.std(flat).cpu().item()
            stats[f"{prefix}_max"] = torch.max(flat).cpu().item()
            stats[f"{prefix}_min"] = torch.min(flat).cpu().item()

            # Enhanced pooling statistics
            stats[f"{prefix}_median"] = torch.median(flat).cpu().item()
            stats[f"{prefix}_q25"] = torch.quantile(flat, 0.25).cpu().item()
            stats[f"{prefix}_q75"] = torch.quantile(flat, 0.75).cpu().item()
            stats[f"{prefix}_iqr"] = (torch.quantile(flat, 0.75) - torch.quantile(flat, 0.25)).cpu().item()

            # Robust statistics
            stats[f"{prefix}_mad"] = torch.median(torch.abs(flat - torch.median(flat))).cpu().item()  # Median Absolute Deviation
            stats[f"{prefix}_range"] = (torch.max(flat) - torch.min(flat)).cpu().item()

            # Statistical moments
            if len(flat) > 1:
                # Skewness approximation
                mean_val = torch.mean(flat)
                std_val = torch.std(flat)
                if std_val > 1e-8:
                    centered = (flat - mean_val) / std_val
                    stats[f"{prefix}_skewness"] = torch.mean(centered ** 3).cpu().item()
                    stats[f"{prefix}_kurtosis"] = torch.mean(centered ** 4).cpu().item() - 3.0
                else:
                    stats[f"{prefix}_skewness"] = 0.0
                    stats[f"{prefix}_kurtosis"] = 0.0
        else:
            # Multi-dimensional pooling across time axis (usually axis=1)
            for i in range(min(tensor.shape[0], 10)):
                dim_data = tensor[i]
                if dim_data.dim() > 1:
                    # Temporal pooling operations
                    dim_data_clean = torch.nan_to_num(dim_data, nan=0.0, posinf=0.0, neginf=0.0)

                    # Mean pooling across time
                    stats[f"{prefix}_{i}_mean_pool"] = torch.mean(dim_data_clean).cpu().item()
                    # Max pooling across time
                    stats[f"{prefix}_{i}_max_pool"] = torch.max(dim_data_clean).cpu().item()
                    # Standard deviation pooling
                    stats[f"{prefix}_{i}_std_pool"] = torch.std(dim_data_clean).cpu().item()

                    # Advanced pooling
                    if dim_data_clean.shape[1] > 1:  # If we have time dimension
                        # Temporal mean and std across frequency bins
                        temporal_mean = torch.mean(dim_data_clean, dim=1)
                        temporal_std = torch.std(dim_data_clean, dim=1)
                        stats[f"{prefix}_{i}_temporal_mean"] = torch.mean(temporal_mean).cpu().item()
                        stats[f"{prefix}_{i}_temporal_std_mean"] = torch.mean(temporal_std).cpu().item()

                        # Frequency mean and std across time
                        freq_mean = torch.mean(dim_data_clean, dim=0)
                        freq_std = torch.std(dim_data_clean, dim=0)
                        stats[f"{prefix}_{i}_freq_mean"] = torch.mean(freq_mean).cpu().item()
                        stats[f"{prefix}_{i}_freq_std_mean"] = torch.mean(freq_std).cpu().item()
                else:
                    # 1D case
                    dim_data_clean = torch.nan_to_num(dim_data, nan=0.0, posinf=0.0, neginf=0.0)
                    stats[f"{prefix}_{i}_mean"] = torch.mean(dim_data_clean).cpu().item()
                    stats[f"{prefix}_{i}_std"] = torch.std(dim_data_clean).cpu().item()

        return stats

    def _compute_spectral_contrast(self, mel_spec):
        """Compute spectral contrast features"""
        n_bands = 6
        band_size = mel_spec.shape[0] // n_bands
        contrast = []

        for i in range(n_bands):
            start = i * band_size
            end = start + band_size
            if end > mel_spec.shape[0]:
                end = mel_spec.shape[0]

            band = mel_spec[start:end, :]
            if band.shape[0] > 0:
                peak = torch.quantile(band, 0.85, dim=0)
                valley = torch.quantile(band, 0.15, dim=0)
                band_contrast = torch.log(peak / (valley + 1e-8) + 1e-8)
                contrast.append(band_contrast)

        if contrast:
            return torch.stack(contrast)
        else:
            return torch.zeros((1, mel_spec.shape[1]), device=mel_spec.device)

    def _extract_label_from_path(self, file_path):
        """Extract artist name from file path"""
        parts = file_path.replace("\\", "/").split("/")
        try:
            train_val_idx = parts.index("train_val")
            return parts[train_val_idx + 1]
        except (ValueError, IndexError):
            return parts[2] if len(parts) > 2 else "unknown"


def load_augmented_data_from_manifest(manifest_path, audio_dir, batch_size=100):
    """
    Load augmented data from saved manifest file with memory-efficient batching

    Args:
        manifest_path: Path to manifest JSON file
        audio_dir: Directory containing audio files
        batch_size: Number of files to load in memory at once

    Returns:
        List of augmented data dictionaries (file paths + metadata, not loaded audio)
    """
    print(f"Loading manifest from {manifest_path}...")

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    # Instead of loading all audio into memory, just prepare file paths
    augmented_data = []

    print(f"Processing {len(manifest)} manifest entries...")

    for entry in tqdm(manifest, desc="Preparing augmented data paths"):
        try:
            audio_path = os.path.join(audio_dir, entry["file"])
            if os.path.exists(audio_path):
                augmented_data.append(
                    {
                        "file_path": audio_path,  # Store path instead of waveform
                        "label": entry["label"],
                        "augmentation_type": entry["augmentation_type"],
                        "source_file": entry.get("source_file", "unknown"),
                    }
                )
        except Exception as e:
            print(f"Failed to prepare {entry['file']}: {e}")
            continue

    print(f"Prepared {len(augmented_data)} augmented sample paths")
    return augmented_data


def load_augmented_batch_lazy(augmented_data_paths, start_idx, batch_size):
    """
    Lazily load a batch of augmented audio data

    Args:
        augmented_data_paths: List of data dictionaries with file_path
        start_idx: Starting index for batch
        batch_size: Size of batch to load

    Returns:
        List of augmented data with loaded waveforms
    """
    batch_data = []
    end_idx = min(start_idx + batch_size, len(augmented_data_paths))

    for i in range(start_idx, end_idx):
        entry = augmented_data_paths[i]
        try:
            waveform, sr = torchaudio.load(entry["file_path"])

            # Ensure 16kHz mono
            if sr != 16000:
                resampler = T.Resample(sr, 16000)
                waveform = resampler(waveform)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            batch_data.append(
                {
                    "waveform": waveform,
                    "label": entry["label"],
                    "augmentation_type": entry["augmentation_type"],
                    "source_file": entry["source_file"],
                }
            )
        except Exception as e:
            print(f"Failed to load {entry['file_path']}: {e}")
            continue

    return batch_data


def extract_features_traditional_augmentation(
    file_list, base_dir="data/artist20/", augmentation_factor=2, batch_size=8
):
    """
    Extract features using traditional augmentation (faster)

    Args:
        file_list: List of audio file paths
        base_dir: Base directory for audio files
        augmentation_factor: Number of augmented versions per original
        batch_size: Batch size for processing

    Returns:
        Features and labels arrays
    """
    print("Using traditional augmentation (faster)...")

    # Prepare full paths
    full_paths = []
    for file_path in file_list:
        clean_path = file_path.lstrip("./")
        full_path = os.path.join(base_dir, clean_path)
        if os.path.exists(full_path):
            full_paths.append(full_path)

    # Initialize extractor with traditional augmentation
    extractor = AugmentedAudioFeatureExtractor(
        use_augmentation=True, augmentation_factor=augmentation_factor
    )

    # Extract features with on-the-fly augmentation
    features, labels = extractor.extract_features_with_augmentation(
        full_paths, batch_size
    )

    return features, labels


def main():
    """Main feature extraction pipeline for augmented data"""
    print("🎵 Feature Extraction from Augmented Data")
    print("=" * 50)

    # Check if augmented data exists
    augmented_dir = "data/artist20_augmented"
    train_manifest_path = os.path.join(augmented_dir, "train_manifest.json")
    val_manifest_path = os.path.join(augmented_dir, "val_manifest.json")

    if os.path.exists(train_manifest_path) and os.path.exists(val_manifest_path):
        print("✅ Found pre-augmented data")
        import sys

        if sys.stdin.isatty():
            use_augmented_input = input(
                "Use pre-augmented audio files? (y/n, default=y): "
            ).lower()
            use_augmented_files = (
                use_augmented_input.startswith("y") if use_augmented_input else True
            )
        else:
            # Non-interactive mode - check if validation data exists and is not empty
            if (
                os.path.getsize(val_manifest_path) <= 2
            ):  # Empty or nearly empty manifest
                print("⚠️  Validation manifest is empty, using on-the-fly augmentation")
                use_augmented_files = False
                use_on_the_fly = True
            else:
                print("🔄 Using pre-augmented data in non-interactive mode")
                use_augmented_files = True

        if use_augmented_files:
            print("\n📂 Loading augmented data from disk...")

            # Load augmented training data (lazy loading for memory efficiency)
            train_augmented_data = load_augmented_data_from_manifest(
                train_manifest_path, augmented_dir
            )

            # Load augmented validation data (lazy loading for memory efficiency)
            val_augmented_data = load_augmented_data_from_manifest(
                val_manifest_path, augmented_dir
            )

            # Extract features from augmented data
            print("Extracting features from training data...")
            extractor = AugmentedAudioFeatureExtractor(use_augmentation=False)
            X_train, y_train = extractor.extract_features_from_augmented_data(
                train_augmented_data
            )

            print("Extracting features from validation data...")
            X_val, y_val = extractor.extract_features_from_augmented_data(
                val_augmented_data
            )
        else:
            # Fall back to on-the-fly augmentation
            print("\n⚡ Using on-the-fly augmentation...")
            use_on_the_fly = True
    else:
        print("❌ No pre-augmented data found")
        print("Please run task1_augmentation.py first, or use on-the-fly augmentation")
        import sys

        if sys.stdin.isatty():
            use_on_the_fly_input = input(
                "Use on-the-fly augmentation? (y/n, default=y): "
            ).lower()
            use_on_the_fly = (
                use_on_the_fly_input.startswith("y") if use_on_the_fly_input else True
            )
        else:
            print("🔄 Using on-the-fly augmentation in non-interactive mode")
            use_on_the_fly = True

    if "use_on_the_fly" in locals() and use_on_the_fly:
        # Load original dataset
        with open("data/artist20/train.json", "r") as f:
            train_files = json.load(f)
        with open("data/artist20/val.json", "r") as f:
            val_files = json.load(f)

        print(f"Original training files: {len(train_files)}")
        print(f"Original validation files: {len(val_files)}")

        print("\n⚡ Extracting features with on-the-fly augmentation...")

        # Extract features with traditional augmentation
        print("Processing training features...")
        X_train, y_train = extract_features_traditional_augmentation(
            train_files, augmentation_factor=3
        )

        print("Processing validation features...")
        X_val, y_val = extract_features_traditional_augmentation(
            val_files, augmentation_factor=1
        )

    print("\nFeature extraction completed:")
    print(f"  Training features shape: {X_train.shape}")
    print(f"  Validation features shape: {X_val.shape}")

    # Save extracted features
    os.makedirs("results/task1_augmented", exist_ok=True)

    np.save("results/task1_augmented/X_train.npy", X_train)
    np.save("results/task1_augmented/y_train.npy", y_train)
    np.save("results/task1_augmented/X_val.npy", X_val)
    np.save("results/task1_augmented/y_val.npy", y_val)

    # Create label mapping
    unique_labels = np.unique(np.concatenate([y_train, y_val]))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}

    with open("results/task1_augmented/label_mapping.json", "w") as f:
        json.dump(label_to_idx, f, indent=2)

    print("\n✅ Features saved to results/task1_augmented/")
    print("📊 Dataset statistics:")
    print(f"  Training samples: {len(y_train)}")
    print(f"  Validation samples: {len(y_val)}")
    print(f"  Number of artists: {len(unique_labels)}")
    if len(X_train) > 0:
        print(f"  Feature vector dimension: {X_train.shape[1]}")
    elif len(X_val) > 0:
        print(f"  Feature vector dimension: {X_val.shape[1]}")
    else:
        print("  Feature vector dimension: unknown (no data)")

    print("\n🔄 Next step:")
    print("  Run: python task1_train_augmented.py")


if __name__ == "__main__":
    main()
