"""Music retrieval system using audio embeddings and cosine similarity."""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import json


class MusicRetrieval:
    """Music retrieval system for finding similar reference tracks."""

    def __init__(self, encoder, reference_dir: str, cache_dir: str = "cache"):
        """
        Initialize music retrieval system.

        Args:
            encoder: Audio encoder instance (CLAP, Stable Audio, etc.)
            reference_dir: Directory containing reference music files
            cache_dir: Directory to cache embeddings
        """
        self.encoder = encoder
        self.reference_dir = Path(reference_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # Storage for reference embeddings
        self.reference_files = []
        self.reference_embeddings = []

        # Load or compute reference embeddings
        self._load_reference_embeddings()

    def _get_cache_path(self, encoder_name: str) -> Path:
        """Get cache file path for embeddings."""
        return self.cache_dir / f"embeddings_{encoder_name}.npz"

    def _load_reference_embeddings(self):
        """Load or compute embeddings for all reference music files."""
        encoder_name = self.encoder.__class__.__name__

        # Get all audio files in reference directory
        audio_extensions = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}
        self.reference_files = sorted(
            [
                f
                for f in self.reference_dir.iterdir()
                if f.suffix.lower() in audio_extensions
            ]
        )

        print(f"Found {len(self.reference_files)} reference files")

        # Try to load from cache
        cache_path = self._get_cache_path(encoder_name)
        if cache_path.exists():
            print(f"Loading cached embeddings from {cache_path}")
            data = np.load(cache_path, allow_pickle=True)
            cached_files = data["files"].tolist()

            # Check if cache matches current files
            if cached_files == [str(f) for f in self.reference_files]:
                self.reference_embeddings = data["embeddings"]
                print("Loaded embeddings from cache")
                return

        # Compute embeddings
        print("Computing reference embeddings...")
        self.reference_embeddings = []

        for i, audio_file in enumerate(self.reference_files):
            print(f"Encoding {i + 1}/{len(self.reference_files)}: {audio_file.name}")
            embedding = self.encoder.encode_audio(str(audio_file))
            self.reference_embeddings.append(embedding)

        self.reference_embeddings = np.array(self.reference_embeddings)

        # Save to cache
        np.savez(
            cache_path,
            files=[str(f) for f in self.reference_files],
            embeddings=self.reference_embeddings,
        )
        print(f"Saved embeddings to cache: {cache_path}")

    def retrieve_similar(
        self, target_audio_path: str, top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Retrieve most similar reference tracks to target audio.

        Args:
            target_audio_path: Path to target audio file
            top_k: Number of top similar tracks to return

        Returns:
            List of (reference_file_path, similarity_score) tuples, sorted by similarity
        """
        # Encode target audio
        print(f"Encoding target audio: {target_audio_path}")
        target_embedding = self.encoder.encode_audio(target_audio_path)

        # Compute similarities with all reference tracks
        similarities = []
        for ref_embedding in self.reference_embeddings:
            similarity = self.encoder.cosine_similarity(target_embedding, ref_embedding)
            similarities.append(similarity)

        similarities = np.array(similarities)

        # Get top-k most similar
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = [
            (str(self.reference_files[idx]), similarities[idx]) for idx in top_indices
        ]

        return results

    def retrieve_all_targets(
        self, target_dir: str, top_k: int = 5
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Retrieve similar tracks for all target files.

        Args:
            target_dir: Directory containing target music files
            top_k: Number of top similar tracks per target

        Returns:
            Dictionary mapping target file paths to their retrieval results
        """
        target_dir = Path(target_dir)

        # Get all target files
        audio_extensions = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}
        target_files = sorted(
            [f for f in target_dir.iterdir() if f.suffix.lower() in audio_extensions]
        )

        print(f"\nFound {len(target_files)} target files")

        results = {}
        for i, target_file in enumerate(target_files):
            print(f"\n{'=' * 60}")
            print(f"Processing target {i + 1}/{len(target_files)}: {target_file.name}")
            print(f"{'=' * 60}")

            similar_tracks = self.retrieve_similar(str(target_file), top_k=top_k)

            results[str(target_file)] = similar_tracks

            # Print results
            print(f"\nTop {top_k} similar tracks:")
            for rank, (ref_path, score) in enumerate(similar_tracks, 1):
                ref_name = Path(ref_path).name
                print(f"  {rank}. {ref_name} (similarity: {score:.4f})")

        return results

    def save_results(self, results: Dict, output_path: str):
        """
        Save retrieval results to JSON file.

        Args:
            results: Retrieval results dictionary
            output_path: Output JSON file path
        """
        # Convert to serializable format
        output = {}
        for target_path, similar_tracks in results.items():
            target_name = Path(target_path).name
            output[target_name] = [
                {"reference": Path(ref).name, "similarity": float(score)}
                for ref, score in similar_tracks
            ]

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"\nSaved retrieval results to: {output_path}")
