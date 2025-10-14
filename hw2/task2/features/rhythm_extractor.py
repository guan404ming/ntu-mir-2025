"""Rhythm and tempo extraction from audio."""

import librosa
import numpy as np
from typing import Dict, Any, Optional


class RhythmExtractor:
    """Extract rhythm and tempo information from audio."""

    def __init__(self, sr: int = 24000):
        """
        Initialize rhythm extractor.

        Args:
            sr: Sample rate for audio processing
        """
        self.sr = sr

    def extract_tempo(self, audio_path: str, duration: Optional[float] = None) -> float:
        """
        Extract tempo in BPM.

        Args:
            audio_path: Path to audio file
            duration: Optional duration to trim audio (seconds)

        Returns:
            Tempo in beats per minute (BPM)
        """
        y, sr = librosa.load(audio_path, sr=self.sr, duration=duration)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        return float(tempo)

    def extract_beat_times(
        self, audio_path: str, duration: Optional[float] = None
    ) -> np.ndarray:
        """
        Extract beat timestamps.

        Args:
            audio_path: Path to audio file
            duration: Optional duration to trim audio (seconds)

        Returns:
            Array of beat times in seconds
        """
        y, sr = librosa.load(audio_path, sr=self.sr, duration=duration)
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beats, sr=sr)
        return beat_times

    def extract_onset_strength(
        self, audio_path: str, duration: Optional[float] = None
    ) -> np.ndarray:
        """
        Extract onset strength envelope for rhythm analysis.

        Args:
            audio_path: Path to audio file
            duration: Optional duration to trim audio (seconds)

        Returns:
            Onset strength envelope
        """
        y, sr = librosa.load(audio_path, sr=self.sr, duration=duration)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        return onset_env

    def extract_rhythm_features(
        self, audio_path: str, duration: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Extract comprehensive rhythm features.

        Args:
            audio_path: Path to audio file
            duration: Optional duration to trim audio (seconds)

        Returns:
            Dictionary of rhythm features
        """
        tempo = self.extract_tempo(audio_path, duration)
        beat_times = self.extract_beat_times(audio_path, duration)

        return {
            "tempo": tempo,
            "beat_times": beat_times.tolist(),
            "num_beats": len(beat_times),
        }
