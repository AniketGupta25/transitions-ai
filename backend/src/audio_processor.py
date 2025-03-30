import os
import uuid
import math
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from pydub import AudioSegment
import librosa
from sklearn.preprocessing import MinMaxScaler


class AudioProcessor:
    def __init__(
        self,
        input_dir: str = "./input",
        output_dir: str = "./output",
        temp_dir: str = "./temp",
    ):
        """Initialize the AudioProcessor with directory paths."""
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.temp_dir = temp_dir
        self._ensure_directories()

    def _ensure_directories(self):
        """Create necessary directories if they don't exist."""
        for directory in [self.input_dir, self.output_dir, self.temp_dir]:
            os.makedirs(directory, exist_ok=True)

    def load_audio(self, file_path: str) -> Tuple[AudioSegment, Dict]:
        """Load an audio file and return AudioSegment with metadata."""
        audio = AudioSegment.from_file(file_path)

        # Normalize volume to -15 dBFS
        target_dBFS = -15.0
        change_in_dBFS = target_dBFS - audio.dBFS
        normalized_audio = audio.apply_gain(change_in_dBFS)

        metadata = {
            "filename": os.path.basename(file_path),
            "duration_ms": len(audio),
            "original_path": file_path,
            "sample_rate": audio.frame_rate,
        }

        return normalized_audio, metadata

    def segment_track(
        self,
        audio: AudioSegment,
        metadata: Dict,
        target_length_ms: int = 90000,
        overlap_ms: int = 2000,
    ) -> List[Dict]:
        """Split audio track into beat-aligned segments."""
        # Convert to numpy array for beat detection
        y, sr = self._audio_segment_to_numpy(audio)

        # Detect beats
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        beat_ms = [int(t * 1000) for t in beat_times]

        segments = []
        current_pos = 0

        while current_pos < len(audio):
            # Find nearest beat for segment start
            start_beat = self._find_nearest_beat(current_pos, beat_ms)
            end_pos = min(start_beat + target_length_ms, len(audio))

            # Extract segment with overlap
            segment_audio = audio[max(0, start_beat) : end_pos]

            if (
                len(segment_audio) < target_length_ms / 2
            ):  # Skip if segment is too short
                break

            segment_id = str(uuid.uuid4())
            segment = {
                "segment_id": segment_id,
                "audio": segment_audio,
                "start_time_ms": start_beat,
                "end_time_ms": end_pos,
                "original_track": metadata["filename"],
                "features": None,  # Will be populated later
            }

            segments.append(segment)
            current_pos = end_pos - overlap_ms

        return segments

    def extract_features(self, segment: Dict) -> Dict:
        """Extract audio features from a segment."""
        y, sr = self._audio_segment_to_numpy(segment["audio"])

        # Basic features
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=y))
        rms = np.mean(librosa.feature.rms(y=y))
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)

        features = {
            "tempo": float(tempo),
            "chroma": chroma,
            "spectral_centroid": float(spectral_centroid),
            "zero_crossing_rate": float(zero_crossing_rate),
            "rms": float(rms),
            "mfcc": mfcc,
        }

        return features

    def calculate_transition_scores(
        self, segments: List[Dict]
    ) -> Tuple[Dict[str, List[Tuple[str, float]]], Dict[str, List[Tuple[str, float]]]]:
        """Calculate transition compatibility scores between segments.

        Returns:
            Tuple containing:
            - transition_scores: Dict mapping segment_id to list of (target_id, score) tuples
            - similarity_scores: Dict mapping segment_id to list of (similar_id, similarity) tuples
        """
        transition_scores = {}
        similarity_scores = {}

        # First calculate similarity scores between segments of the same track
        for source in segments:
            similarity_scores[source["segment_id"]] = []
            source_features = source["features"]

            # Compare with other segments from same track
            for target in segments:
                target_features = target["features"]
                if (
                    source["original_track"] == target["original_track"]
                    and source["segment_id"] != target["segment_id"]
                ):
                    try:
                        similarity = self._calculate_similarity_score(
                            source_features, target_features
                        )
                        similarity_scores[source["segment_id"]].append(
                            (target["segment_id"], similarity)
                        )
                    except Exception as e:
                        print(f"⚠️ Error calculating similarity: {str(e)}")
                        continue

        # Calculate transition scores
        for source in segments:
            transition_scores[source["segment_id"]] = []
            source_features = source["features"]

            for target in segments:
                if source["original_track"] == target["original_track"]:
                    continue

                target_features = target["features"]
                try:
                    score = self._calculate_compatibility_score(
                        source_features, target_features
                    )

                    if score > 50:  # Only keep good transitions
                        transition_scores[source["segment_id"]].append(
                            (target["segment_id"], score)
                        )
                except Exception as e:
                    print(f"⚠️ Error calculating transition score: {str(e)}")
                    continue

            # Sort by score and keep top 5
            transition_scores[source["segment_id"]].sort(
                key=lambda x: x[1], reverse=True
            )
            transition_scores[source["segment_id"]] = transition_scores[
                source["segment_id"]
            ][:5]

        return transition_scores, similarity_scores

    def create_transition(
        self,
        source_segment: Dict,
        target_segment: Dict,
        crossfade_duration_ms: int = 3000,
    ) -> AudioSegment:
        """Create a crossfaded transition between two segments."""
        # Calculate transition position
        transition_pos = len(source_segment["audio"]) - crossfade_duration_ms

        # Create crossfade with fixed gain
        mixed = source_segment["audio"].overlay(
            target_segment["audio"],
            position=transition_pos,
            gain_during_overlay=-3,  # Fixed gain value in dB
        )

        return mixed

    def _audio_segment_to_numpy(
        self, audio_segment: AudioSegment
    ) -> Tuple[np.ndarray, int]:
        """Convert AudioSegment to numpy array for librosa processing."""
        # Export to temp file
        temp_path = os.path.join(self.temp_dir, f"temp_{uuid.uuid4()}.wav")
        audio_segment.export(temp_path, format="wav")

        # Load with librosa
        y, sr = librosa.load(temp_path, sr=None)

        # Clean up
        os.remove(temp_path)

        return y, sr

    def _find_nearest_beat(self, position_ms: int, beat_times_ms: List[int]) -> int:
        """Find the nearest beat time to a given position."""
        return min(beat_times_ms, key=lambda x: abs(x - position_ms))

    def _calculate_compatibility_score(
        self, source_features: Dict, target_features: Dict
    ) -> float:
        """Calculate compatibility score between two segments based on their features."""
        # Tempo compatibility (0-100)
        tempo_diff = abs(source_features["tempo"] - target_features["tempo"])
        tempo_score = 100 - min(100, tempo_diff)

        # Harmonic compatibility using chroma features (0-100)
        chroma_similarity = np.dot(
            source_features["chroma"], target_features["chroma"]
        ) / (
            np.linalg.norm(source_features["chroma"])
            * np.linalg.norm(target_features["chroma"])
        )
        chroma_score = float(chroma_similarity * 100)

        # Timbre compatibility using MFCCs (0-100)
        mfcc_similarity = np.dot(source_features["mfcc"], target_features["mfcc"]) / (
            np.linalg.norm(source_features["mfcc"])
            * np.linalg.norm(target_features["mfcc"])
        )
        timbre_score = float(mfcc_similarity * 100)

        # Energy compatibility (0-100)
        energy_diff = abs(source_features["rms"] - target_features["rms"])
        energy_score = 100 - min(100, energy_diff * 100)

        # Weighted average of scores
        weights = {"tempo": 0.4, "chroma": 0.3, "timbre": 0.2, "energy": 0.1}

        final_score = (
            tempo_score * weights["tempo"]
            + chroma_score * weights["chroma"]
            + timbre_score * weights["timbre"]
            + energy_score * weights["energy"]
        )

        return final_score

    def _calculate_similarity_score(
        self, source_features: Dict, target_features: Dict
    ) -> float:
        """Calculate similarity score between two segments (0-100)."""
        # 1. Compare multiple features with different weights
        feature_similarities = {
            # Timbre comparison (MFCCs)
            "mfcc": np.dot(source_features["mfcc"], target_features["mfcc"])
            / (
                np.linalg.norm(source_features["mfcc"])
                * np.linalg.norm(target_features["mfcc"])
            ),
            # Harmonic content (Chroma)
            "chroma": np.dot(source_features["chroma"], target_features["chroma"])
            / (
                np.linalg.norm(source_features["chroma"])
                * np.linalg.norm(target_features["chroma"])
            ),
            # Energy difference
            "energy": 1
            - abs(source_features["rms"] - target_features["rms"])
            / max(source_features["rms"], target_features["rms"]),
            # Tempo similarity
            "tempo": 1
            - abs(source_features["tempo"] - target_features["tempo"])
            / max(source_features["tempo"], 1),
        }

        # 2. Apply stricter weights that require multiple features to be similar
        weights = {
            "mfcc": 0.4,  # Timbre similarity
            "chroma": 0.3,  # Harmonic similarity
            "energy": 0.2,  # Energy level similarity
            "tempo": 0.1,  # Tempo similarity
        }

        # 3. Apply non-linear scaling to make the similarity score more discriminative
        similarity = sum(
            weights[feature]
            * (score**2)  # Square the scores to make them more selective
            for feature, score in feature_similarities.items()
        )

        # 4. Apply exponential scaling to make high similarities harder to achieve
        final_score = (
            similarity**1.5
        ) * 100  # Exponential scaling makes it harder to get very high scores

        return float(final_score)
