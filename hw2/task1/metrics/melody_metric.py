"""Melody accuracy metric using chromagram-based one-hot encoding."""

import torchaudio
import numpy as np
import librosa
import scipy.signal as signal
from torchaudio import transforms as T
from pathlib import Path


class MelodyMetric:
    """Calculate melody accuracy between audio pairs."""

    def __init__(self, sr=44100, cutoff=261.2, win_length=2048, hop_length=256):
        """
        Initialize melody metric.

        Args:
            sr: Sample rate for audio processing
            cutoff: High-pass filter cutoff frequency (Hz)
            win_length: STFT window length
            hop_length: STFT hop length
        """
        self.sr = sr
        self.cutoff = cutoff
        self.win_length = win_length
        self.hop_length = hop_length

    def extract_melody_one_hot(self, audio_path: str) -> np.ndarray:
        """
        Extract one-hot chromagram-based melody from audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            One-hot chromagram array of shape (12, n_frames)
        """
        # Load audio
        audio, in_sr = torchaudio.load(audio_path)

        # Convert to mono
        audio_mono = audio.mean(dim=0)

        # Resample if necessary
        if in_sr != self.sr:
            resample_tf = T.Resample(orig_freq=in_sr, new_freq=self.sr)
            audio_mono = resample_tf(audio_mono)

        # Convert to numpy
        y = audio_mono.numpy()

        # Apply high-pass filter
        nyquist = 0.5 * self.sr
        norm_cutoff = self.cutoff / nyquist
        b, a = signal.butter(N=2, Wn=norm_cutoff, btype="high", analog=False)
        y_hp = signal.filtfilt(b, a, y)

        # Compute chromagram
        chroma = librosa.feature.chroma_stft(
            y=y_hp,
            sr=self.sr,
            n_fft=self.win_length,
            win_length=self.win_length,
            hop_length=self.hop_length,
        )

        # Convert to one-hot
        pitch_class_idx = np.argmax(chroma, axis=0)
        one_hot_chroma = np.zeros_like(chroma)
        one_hot_chroma[pitch_class_idx, np.arange(chroma.shape[1])] = 1.0

        return one_hot_chroma

    def calculate_accuracy(
        self, target_path: str, generated_path: str, target_duration: float = None
    ) -> float:
        """
        Calculate melody accuracy between target and generated audio.

        Args:
            target_path: Path to target/ground truth audio
            generated_path: Path to generated audio
            target_duration: If specified, trim target to this duration (seconds)

        Returns:
            Melody accuracy score (0-1)
        """
        # Extract melodies
        gt_melody = self.extract_melody_one_hot(target_path)
        gen_melody = self.extract_melody_one_hot(generated_path)

        # Trim target if duration specified
        if target_duration is not None:
            max_frames = int(target_duration * self.sr / self.hop_length)
            gt_melody = gt_melody[:, :max_frames]

        # Compare up to minimum length
        min_len = min(gen_melody.shape[1], gt_melody.shape[1])

        # Count matching frames (where both have same active pitch class)
        matches = (
            (gen_melody[:, :min_len] == gt_melody[:, :min_len])
            & (gen_melody[:, :min_len] == 1)
        ).sum()

        accuracy = matches / min_len

        return float(accuracy)

    def evaluate(
        self, target_path: str, generated_path: str, target_duration: float = None
    ) -> dict:
        """
        Evaluate melody similarity with detailed results.

        Args:
            target_path: Path to target audio
            generated_path: Path to generated audio
            target_duration: If specified, trim target to this duration

        Returns:
            Dictionary with evaluation results
        """
        accuracy = self.calculate_accuracy(target_path, generated_path, target_duration)

        return {
            "melody_accuracy": accuracy,
            "target": Path(target_path).name,
            "generated": Path(generated_path).name,
            "target_duration": target_duration,
        }
