"""Chord extraction from audio."""

import librosa
import numpy as np
from typing import List, Tuple, Optional


class ChordExtractor:
    """Extract chord information from audio."""

    def __init__(self, sr: int = 24000):
        """
        Initialize chord extractor.

        Args:
            sr: Sample rate for audio processing
        """
        self.sr = sr

        # Major and minor chord templates (simplified)
        self.major_template = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0])
        self.minor_template = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0])

    def extract_chords(
        self, audio_path: str, duration: Optional[float] = None
    ) -> List[Tuple[float, str]]:
        """
        Extract chord progression from audio.

        Args:
            audio_path: Path to audio file
            duration: Optional duration to trim audio (seconds)

        Returns:
            List of (time, chord_name) tuples
        """
        # Load audio
        y, sr = librosa.load(audio_path, sr=self.sr, duration=duration)

        # Extract chroma
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)

        # Simple chord recognition based on chroma templates
        hop_length = 512
        times = librosa.frames_to_time(
            np.arange(chroma.shape[1]), sr=sr, hop_length=hop_length
        )

        chords = []
        chord_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

        for i in range(chroma.shape[1]):
            frame = chroma[:, i]

            # Try all possible roots for major and minor
            max_correlation = -1
            best_chord = "N"  # No chord

            for root in range(12):
                # Check major
                major = np.roll(self.major_template, root)
                corr = np.corrcoef(frame, major)[0, 1]
                if corr > max_correlation:
                    max_correlation = corr
                    best_chord = chord_names[root]

                # Check minor
                minor = np.roll(self.minor_template, root)
                corr = np.corrcoef(frame, minor)[0, 1]
                if corr > max_correlation:
                    max_correlation = corr
                    best_chord = chord_names[root] + "m"

            chords.append((times[i], best_chord))

        # Simplify by grouping consecutive identical chords
        simplified_chords = []
        if chords:
            current_chord = chords[0][1]
            current_time = chords[0][0]

            for time, chord in chords[1:]:
                if chord != current_chord:
                    simplified_chords.append((current_time, current_chord))
                    current_chord = chord
                    current_time = time

            simplified_chords.append((current_time, current_chord))

        return simplified_chords

    def extract_chord_sequence(
        self, audio_path: str, duration: Optional[float] = None
    ) -> str:
        """
        Extract chord progression as a string sequence.

        Args:
            audio_path: Path to audio file
            duration: Optional duration to trim audio (seconds)

        Returns:
            Space-separated chord sequence
        """
        chords = self.extract_chords(audio_path, duration)
        chord_names = [chord for _, chord in chords]
        return " ".join(chord_names)
