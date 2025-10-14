"""Melody extraction from audio using librosa."""

import librosa
import numpy as np
from typing import Tuple, Optional


class MelodyExtractor:
    """Extract melody (pitch) information from audio."""

    def __init__(self, sr: int = 24000):
        """
        Initialize melody extractor.

        Args:
            sr: Sample rate for audio processing
        """
        self.sr = sr

    def extract_melody(
        self, audio_path: str, duration: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract melody as pitch contour from audio.

        Args:
            audio_path: Path to audio file
            duration: Optional duration to trim audio (seconds)

        Returns:
            Tuple of (times, pitches) where:
            - times: Time stamps in seconds
            - pitches: Pitch values in Hz (0 for unvoiced)
        """
        # Load audio
        y, sr = librosa.load(audio_path, sr=self.sr, duration=duration)

        # Extract pitch using piptrack
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr, fmin=50, fmax=2000)

        # Select pitch with highest magnitude at each frame
        pitch_contour = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            pitch_contour.append(pitch)

        pitch_contour = np.array(pitch_contour)

        # Generate time stamps
        hop_length = 512  # default hop length
        times = librosa.frames_to_time(
            np.arange(len(pitch_contour)), sr=sr, hop_length=hop_length
        )

        return times, pitch_contour

    def extract_chroma(
        self, audio_path: str, duration: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract chromagram representation.

        Args:
            audio_path: Path to audio file
            duration: Optional duration to trim audio (seconds)

        Returns:
            Tuple of (times, chroma) where:
            - times: Time stamps in seconds
            - chroma: Chroma features (12 x frames)
        """
        # Load audio
        y, sr = librosa.load(audio_path, sr=self.sr, duration=duration)

        # Extract chroma
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)

        # Generate time stamps
        hop_length = 512
        times = librosa.frames_to_time(
            np.arange(chroma.shape[1]), sr=sr, hop_length=hop_length
        )

        return times, chroma

    def extract_melody_for_musicgen(
        self, audio_path: str, duration: Optional[float] = None
    ) -> np.ndarray:
        """
        Extract melody in format suitable for MusicGen conditioning.

        MusicGen expects a melody as a single-channel audio tensor.
        We use the predominant pitch to create a simple melody representation.

        Args:
            audio_path: Path to audio file
            duration: Optional duration to trim audio (seconds)

        Returns:
            Melody audio as numpy array (mono)
        """
        # Simply return the audio itself - MusicGen will use it as melody guide
        # According to rules, we can extract melody using MIR tools
        y, sr = librosa.load(audio_path, sr=self.sr, duration=duration, mono=True)

        # Apply harmonic-percussive separation to get melodic content
        y_harmonic, y_percussive = librosa.effects.hpss(y)

        # Return harmonic component as melody
        return y_harmonic
