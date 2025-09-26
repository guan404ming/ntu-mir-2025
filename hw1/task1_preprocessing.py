import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
import json
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class AudioFeatureExtractor:
    def __init__(self, sr=16000):
        self.sr = sr
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.mfcc_transform = T.MFCC(
            sample_rate=sr,
            n_mfcc=20,
            melkwargs={'n_fft': 2048, 'hop_length': 512, 'n_mels': 128}
        ).to(self.device)

        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sr,
            n_fft=2048,
            hop_length=512,
            n_mels=128
        ).to(self.device)

        self.spectral_centroid = T.SpectralCentroid(
            sample_rate=sr,
            n_fft=2048,
            hop_length=512
        ).to(self.device)

    def extract_features_batch(self, audio_paths, batch_size=8):
        all_features = []
        all_labels = []

        with tqdm(total=len(audio_paths), desc="Extracting features", unit="file", ncols=100, colour="green") as pbar:
            for i in range(0, len(audio_paths), batch_size):
                batch_paths = audio_paths[i:i+batch_size]
                batch_features, batch_labels = self._process_batch(batch_paths)
                all_features.extend(batch_features)
                all_labels.extend(batch_labels)
                pbar.update(len(batch_paths))
                pbar.set_postfix({"success": len(all_features), "failed": len(batch_paths) - len(batch_features)})

        if len(all_features) != len(audio_paths):
            failed_count = len(audio_paths) - len(all_features)
            print(f"\nWarning: Failed to process {failed_count} files")

        return np.array(all_features), np.array(all_labels)

    def _process_batch(self, batch_paths):
        batch_features = []
        batch_labels = []

        with torch.no_grad():
            for audio_path in batch_paths:
                try:
                    waveform, orig_sr = torchaudio.load(audio_path)

                    if orig_sr != self.sr:
                        resampler = T.Resample(orig_sr, self.sr)
                        waveform = resampler(waveform)

                    if waveform.shape[0] > 1:
                        waveform = torch.mean(waveform, dim=0, keepdim=True)

                    waveform = waveform.to(self.device)
                    features = self._extract_single_features(waveform)

                    if features is not None:
                        batch_features.append(features)
                        batch_labels.append(self._extract_label_from_path(audio_path))

                except Exception:
                    continue

        return batch_features, batch_labels

    def _extract_single_features(self, waveform):
        try:
            features = {}

            mfccs = self.mfcc_transform(waveform)
            features.update(self._compute_statistics(mfccs[0], 'mfcc'))

            mel_spec = self.mel_spectrogram(waveform)
            features.update(self._compute_statistics(mel_spec[0], 'mel'))

            chroma_like = mel_spec[0, :12, :]
            features.update(self._compute_statistics(chroma_like, 'chroma'))

            centroid = self.spectral_centroid(waveform)
            features.update(self._compute_statistics(centroid[0], 'centroid'))

            rms = torch.sqrt(torch.mean(waveform**2, dim=-1, keepdim=True))
            features.update(self._compute_statistics(rms[0], 'rms'))

            sign_changes = torch.abs(torch.diff(torch.sign(waveform), dim=-1))
            zcr = torch.mean(sign_changes, dim=-1, keepdim=True)
            features.update(self._compute_statistics(zcr[0], 'zcr'))

            mel_energy = torch.sum(mel_spec[0], dim=0, keepdim=True)
            features.update(self._compute_statistics(mel_energy, 'rolloff'))

            if waveform.shape[-1] > self.sr:
                diff_spec = torch.diff(mel_spec[0], dim=1)
                onset_strength = torch.mean(torch.clamp(diff_spec, min=0), dim=0)
                features['tempo'] = torch.mean(onset_strength).cpu().item() * 100 + 60

            freq_weighted = torch.arange(mel_spec.shape[1], dtype=torch.float32, device=self.device).unsqueeze(1)
            bandwidth = torch.sum(freq_weighted * mel_spec[0], dim=0) / (torch.sum(mel_spec[0], dim=0) + 1e-8)
            features.update(self._compute_statistics(bandwidth.unsqueeze(0), 'bandwidth'))

            # Add delta MFCC features (velocity)
            mfcc_delta = torch.diff(mfccs[0], dim=1, prepend=mfccs[0, :, :1])
            features.update(self._compute_statistics(mfcc_delta, 'mfcc_delta'))

            # Add delta-delta MFCC features (acceleration)
            mfcc_delta2 = torch.diff(mfcc_delta, dim=1, prepend=mfcc_delta[:, :1])
            features.update(self._compute_statistics(mfcc_delta2, 'mfcc_delta2'))

            # Add spectral contrast features
            contrast = self._compute_spectral_contrast(mel_spec[0])
            features.update(self._compute_statistics(contrast, 'contrast'))

            feature_array = []
            for key in sorted(features.keys()):
                value = features[key]
                if isinstance(value, torch.Tensor):
                    val = value.cpu().item()
                else:
                    val = float(value)

                # Handle NaN, inf values
                if np.isnan(val) or np.isinf(val):
                    val = 0.0

                feature_array.append(val)

            return feature_array

        except Exception:
            return None

    def _compute_statistics(self, tensor, prefix):
        stats = {}

        if tensor.dim() == 0:
            val = tensor.cpu().item()
            stats[f'{prefix}_mean'] = 0.0 if (np.isnan(val) or np.isinf(val)) else val
            return stats

        if tensor.dim() == 1 or (tensor.dim() == 2 and tensor.shape[0] == 1):
            flat = tensor.flatten()

            # Replace NaN/inf values in tensor before computing stats
            flat = torch.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)

            stats[f'{prefix}_mean'] = torch.mean(flat).cpu().item()
            stats[f'{prefix}_std'] = torch.std(flat).cpu().item()
            stats[f'{prefix}_max'] = torch.max(flat).cpu().item()
            stats[f'{prefix}_min'] = torch.min(flat).cpu().item()
        else:
            for i in range(min(tensor.shape[0], 10)):
                dim_data = tensor[i].flatten()

                # Replace NaN/inf values in tensor before computing stats
                dim_data = torch.nan_to_num(dim_data, nan=0.0, posinf=0.0, neginf=0.0)

                stats[f'{prefix}_{i}_mean'] = torch.mean(dim_data).cpu().item()
                stats[f'{prefix}_{i}_std'] = torch.std(dim_data).cpu().item()

        return stats

    def _compute_spectral_contrast(self, mel_spec):
        """Compute spectral contrast features"""
        # Divide spectrum into frequency bands
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
                # Compute ratio of peak to valley
                peak = torch.quantile(band, 0.85, dim=0)
                valley = torch.quantile(band, 0.15, dim=0)
                band_contrast = torch.log(peak / (valley + 1e-8) + 1e-8)
                contrast.append(band_contrast)

        if contrast:
            return torch.stack(contrast)
        else:
            return torch.zeros((1, mel_spec.shape[1]), device=mel_spec.device)

    def _extract_label_from_path(self, file_path):
        # Extract artist name from path like: ./train_val/aerosmith/album/song.mp3
        # or full path like: data/artist20/train_val/aerosmith/album/song.mp3
        parts = file_path.replace('\\', '/').split('/')

        # Find 'train_val' in the path and get the next part (artist name)
        try:
            train_val_idx = parts.index('train_val')
            return parts[train_val_idx + 1]  # Artist name is right after 'train_val'
        except (ValueError, IndexError):
            # Fallback: assume artist is at index 2 for relative paths
            return parts[2] if len(parts) > 2 else 'unknown'

def extract_features_parallel(file_list, base_dir='data/artist20/', batch_size=16):
    if not torch.cuda.is_available():
        batch_size = max(1, batch_size // 4)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    extractor = AudioFeatureExtractor()

    full_paths = []
    for file_path in file_list:
        clean_path = file_path.lstrip('./')
        full_path = os.path.join(base_dir, clean_path)
        if os.path.exists(full_path):
            full_paths.append(full_path)

    print(f"Processing {len(full_paths)} files with batch_size={batch_size}")
    features, labels = extractor.extract_features_batch(full_paths, batch_size=batch_size)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return features, labels

def extract_label_from_path(file_path):
    return file_path.split('/')[2]  # artist name is the 3rd element

def main():
    print("Starting feature extraction for Task 1...")

    # Load training and validation data
    with open('data/artist20/train.json', 'r') as f:
        train_files = json.load(f)

    with open('data/artist20/val.json', 'r') as f:
        val_files = json.load(f)

    print(f"Training files: {len(train_files)}")
    print(f"Validation files: {len(val_files)}")

    # Extract features for training set
    print("\nExtracting training features...")
    X_train, y_train = extract_features_parallel(train_files)

    # Extract features for validation set
    print("\nExtracting validation features...")
    X_val, y_val = extract_features_parallel(val_files)

    print(f"\nTraining features shape: {X_train.shape}")
    print(f"Validation features shape: {X_val.shape}")

    # Save extracted features
    os.makedirs('results/task1', exist_ok=True)

    np.save('results/task1/X_train.npy', X_train)
    np.save('results/task1/y_train.npy', y_train)
    np.save('results/task1/X_val.npy', X_val)
    np.save('results/task1/y_val.npy', y_val)

    # Create label mapping
    unique_labels = np.unique(np.concatenate([y_train, y_val]))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}

    with open('results/task1/label_mapping.json', 'w') as f:
        json.dump(label_to_idx, f, indent=2)

    print(f"\nExtracted features saved to results/task1/")
    print(f"Number of unique artists: {len(unique_labels)}")
    print(f"Feature vector dimension: {X_train.shape[1]}")

if __name__ == "__main__":
    main()