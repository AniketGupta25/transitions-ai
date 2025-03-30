import os
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import soundfile as sf
# from essentia.standard import KeyExtractor  # Commented out problematic dependency
import pickle
# import madmom  # Commented out problematic dependency
from tqdm import tqdm
import traceback

from transitions_ai.src.config import (
    SAMPLE_RATE, HOP_LENGTH, N_FFT, VISUALIZATION_DIR, TEMP_DIR
)
from transitions_ai.src.logger import get_logger, log_audio_analysis

logger = get_logger("audio_analysis")

class AudioAnalyzer:
    """
    Class for analyzing audio files and extracting relevant features for DJ mashups
    """
    
    def __init__(self, file_path: str, resample: bool = True):
        """
        Initialize the analyzer with an audio file
        
        Args:
            file_path: Path to the audio file
            resample: Whether to resample the audio to the config sample rate
        """
        self.file_path = Path(file_path)
        self.track_name = self.file_path.stem
        logger.info(f"Initializing analyzer for {self.track_name}")
        
        # Load audio file
        try:
            self.y, self.sr = librosa.load(file_path, sr=SAMPLE_RATE if resample else None)
            logger.info(f"Loaded audio file: {self.track_name} ({len(self.y)/self.sr:.2f}s)")
        except Exception as e:
            logger.error(f"Error loading audio file: {e}")
            raise
            
        # Initialize feature cache
        self.features = {}
        self.cache_file = TEMP_DIR / f"{self.track_name}_features.pkl"
        
        # Load cached features if available
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    self.features = pickle.load(f)
                logger.info(f"Loaded cached features for {self.track_name}")
            except Exception as e:
                logger.warning(f"Error loading cached features: {e}")
                self.features = {}
    
    def extract_all_features(self) -> Dict[str, Any]:
        """Extract all audio features and return a dictionary of results"""
        logger.info(f"Extracting all features for {self.track_name}")
        
        try:
            # Extract basic features
            self.features['bpm'] = self.get_bpm()
            self.features['key'] = self.get_key()
            self.features['duration'] = float(len(self.y) / self.sr)  # Ensure float type
            
            # Extract rhythm features
            self.features['beat_frames'] = self.get_beats().tolist()  # Convert to list
            self.features['beat_times'] = librosa.frames_to_time(self.features['beat_frames'], sr=self.sr).tolist()  # Convert to list
            self.features['beat_strength'] = self.get_beat_strength().tolist()  # Convert to list
            
            # Extract spectral features
            self.features['spectral_centroid'] = float(np.mean(librosa.feature.spectral_centroid(y=self.y, sr=self.sr)[0]))
            self.features['spectral_rolloff'] = float(np.mean(librosa.feature.spectral_rolloff(y=self.y, sr=self.sr)[0]))
            self.features['spectral_contrast'] = np.mean(librosa.feature.spectral_contrast(y=self.y, sr=self.sr), axis=1).tolist()
            
            # Extract energy and loudness
            self.features['energy_envelope'] = self.get_energy_envelope().tolist()  # Store as list
            self.features['rms_envelope'] = self.get_rms_envelope().tolist()  # Store as list
            self.features['average_loudness'] = float(np.mean(self.features['rms_envelope']))
            
            # Extract harmonic features - store chromagram as 2D list
            chroma = librosa.feature.chroma_stft(y=self.y, sr=self.sr)
            self.features['chroma'] = chroma.tolist()  # Convert to list for storage
            self.features['chroma_variance'] = np.var(chroma, axis=1).tolist()
            
            # Extract onset information
            onset_env = librosa.onset.onset_strength(y=self.y, sr=self.sr)
            self.features['onset_envelope'] = onset_env.tolist()  # Convert to list
            self.features['onset_frames'] = librosa.onset.onset_detect(onset_envelope=onset_env).tolist()
            self.features['onset_times'] = librosa.frames_to_time(self.features['onset_frames'], sr=self.sr).tolist()
            
            # Save features to cache
            try:
                with open(self.cache_file, 'wb') as f:
                    pickle.dump(self.features, f)
                logger.info(f"Cached features for {self.track_name}")
            except Exception as e:
                logger.warning(f"Error caching features: {e}")
            
            # Log the analysis results
            log_audio_analysis(self.track_name, self.features)
            
            return self.features
        except Exception as e:
            logger.error(f"Error in extract_all_features: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def get_bpm(self) -> float:
        """Detect the tempo (BPM) of the track"""
        if 'bpm' in self.features:
            return self.features['bpm']
            
        logger.info(f"Detecting BPM for {self.track_name}")
        
        try:
            # Use librosa's beat tracking with explicit type conversion
            tempo, _ = librosa.beat.beat_track(y=self.y, sr=self.sr)
            tempo_float = float(tempo)  # Ensure it's a Python float, not numpy.float
                
            logger.info(f"Detected BPM: {tempo_float:.2f}")
            return tempo_float
        except Exception as e:
            logger.error(f"Error in BPM detection: {str(e)}")
            logger.error(traceback.format_exc())
            return 120.0  # Default fallback BPM
    
    def get_key(self) -> str:
        """Detect the musical key of the track"""
        if 'key' in self.features:
            return self.features['key']
            
        logger.info(f"Detecting key for {self.track_name}")
            
        try:
            # Use librosa's chroma features for key detection
            chroma = librosa.feature.chroma_stft(y=self.y, sr=self.sr)
            chroma_avg = np.mean(chroma, axis=1)
            key_idx = int(np.argmax(chroma_avg))  # Convert to int
            key_map = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            key = key_map[key_idx]
            
            # Determine major/minor (simplified)
            major_profile = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
            minor_profile = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0])
            
            # Rotate profiles to match key
            major_profile = np.roll(major_profile, key_idx)
            minor_profile = np.roll(minor_profile, key_idx)
            
            # Calculate correlation
            major_corr = float(np.corrcoef(chroma_avg, major_profile)[0, 1])
            minor_corr = float(np.corrcoef(chroma_avg, minor_profile)[0, 1])
            
            scale = "major" if major_corr > minor_corr else "minor"
            key_result = f"{key} {scale}"
            
            logger.info(f"Detected key: {key_result}")
            return key_result
        except Exception as e:
            logger.error(f"Error in key detection: {str(e)}")
            logger.error(traceback.format_exc())
            return "C major"  # Default fallback key
    
    def get_beats(self) -> np.ndarray:
        """Detect beat frames in the track"""
        if 'beat_frames' in self.features:
            if isinstance(self.features['beat_frames'], list):
                return np.array(self.features['beat_frames'])
            return self.features['beat_frames']
            
        logger.info(f"Detecting beats for {self.track_name}")
        
        try:
            # Use librosa for beat tracking
            _, beat_frames = librosa.beat.beat_track(y=self.y, sr=self.sr, hop_length=HOP_LENGTH)
            logger.info(f"Detected {len(beat_frames)} beats")
            return beat_frames
        except Exception as e:
            logger.error(f"Error in beat detection: {str(e)}")
            logger.error(traceback.format_exc())
            # Create a simple fallback - one beat every 0.5 seconds
            fallback_beats = np.arange(0, len(self.y), self.sr // 2)
            return librosa.time_to_frames(librosa.samples_to_time(fallback_beats, sr=self.sr), sr=self.sr)
    
    def get_beat_strength(self) -> np.ndarray:
        """Calculate the strength of each detected beat"""
        if 'beat_strength' in self.features:
            if isinstance(self.features['beat_strength'], list):
                return np.array(self.features['beat_strength'])
            return self.features['beat_strength']
            
        logger.info(f"Calculating beat strength for {self.track_name}")
        
        try:
            # Calculate onset envelope if not already done
            if 'onset_envelope' not in self.features:
                self.features['onset_envelope'] = librosa.onset.onset_strength(y=self.y, sr=self.sr)
            elif isinstance(self.features['onset_envelope'], list):
                self.features['onset_envelope'] = np.array(self.features['onset_envelope'])
            
            # Get beat frames if not already calculated
            if 'beat_frames' not in self.features:
                self.features['beat_frames'] = self.get_beats()
            elif isinstance(self.features['beat_frames'], list):
                self.features['beat_frames'] = np.array(self.features['beat_frames'])
            
            # Get strength of each beat from onset envelope
            beat_frames = self.features['beat_frames']
            beat_frames = np.clip(beat_frames, 0, len(self.features['onset_envelope'])-1)  # Ensure indices are in range
            beat_strength = self.features['onset_envelope'][beat_frames]
            return beat_strength
        except Exception as e:
            logger.error(f"Error in beat strength calculation: {str(e)}")
            logger.error(traceback.format_exc())
            # Create a fallback
            return np.ones(len(self.get_beats()))
    
    def get_energy_envelope(self) -> np.ndarray:
        """Calculate the energy envelope of the track"""
        if 'energy_envelope' in self.features:
            if isinstance(self.features['energy_envelope'], list):
                return np.array(self.features['energy_envelope'])
            return self.features['energy_envelope']
            
        logger.info(f"Calculating energy envelope for {self.track_name}")
        
        try:
            # Square the amplitude
            energy = self.y ** 2
            
            # Apply a moving average filter
            frame_length = int(0.05 * self.sr)  # 50ms window
            energy_envelope = np.convolve(energy, np.ones(frame_length)/frame_length, mode='same')
            
            return energy_envelope
        except Exception as e:
            logger.error(f"Error calculating energy envelope: {str(e)}")
            logger.error(traceback.format_exc())
            # Create a fallback
            return np.ones(len(self.y) // HOP_LENGTH + 1)
    
    def get_rms_envelope(self) -> np.ndarray:
        """Calculate the RMS (loudness) envelope of the track"""
        if 'rms_envelope' in self.features:
            if isinstance(self.features['rms_envelope'], list):
                return np.array(self.features['rms_envelope'])
            return self.features['rms_envelope']
            
        logger.info(f"Calculating RMS envelope for {self.track_name}")
        
        try:
            hop_length = HOP_LENGTH
            rms = librosa.feature.rms(y=self.y, hop_length=hop_length)[0]
            
            return rms
        except Exception as e:
            logger.error(f"Error calculating RMS envelope: {str(e)}")
            logger.error(traceback.format_exc())
            # Create a fallback
            return np.ones(len(self.y) // HOP_LENGTH + 1)
    
    def visualize_features(self, save_path: Optional[Path] = None) -> None:
        """Generate visualizations of the extracted features"""
        if not save_path:
            save_path = VISUALIZATION_DIR / f"{self.track_name}_analysis.png"
        
        logger.info(f"Generating visualizations for {self.track_name}")
        
        try:
            # Ensure all features are extracted
            if len(self.features) == 0:
                self.extract_all_features()
            
            # Convert lists back to numpy arrays for visualization
            beat_times = np.array(self.features['beat_times']) if isinstance(self.features['beat_times'], list) else self.features['beat_times']
            onset_envelope = np.array(self.features['onset_envelope']) if isinstance(self.features['onset_envelope'], list) else self.features['onset_envelope']
            rms_envelope = np.array(self.features['rms_envelope']) if isinstance(self.features['rms_envelope'], list) else self.features['rms_envelope']
            chroma = np.array(self.features['chroma']) if isinstance(self.features['chroma'], list) else self.features['chroma']
            
            # Create figure with multiple subplots
            fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
            
            # Time axis in seconds
            times = librosa.times_like(rms_envelope, sr=self.sr, hop_length=HOP_LENGTH)
            
            # Plot waveform
            librosa.display.waveshow(self.y, sr=self.sr, ax=axes[0])
            axes[0].set_title('Waveform')
            
            # Plot beats and onsets
            axes[1].plot(times, librosa.util.normalize(onset_envelope), label='Onset Strength')
            for beat_time in beat_times:
                axes[1].axvline(beat_time, color='r', alpha=0.5, linestyle='--')
            axes[1].legend()
            axes[1].set_title(f'Beats and Onsets (BPM: {self.features["bpm"]:.1f})')
            
            # Plot RMS energy
            axes[2].plot(times, rms_envelope)
            axes[2].set_title('RMS Energy (Loudness)')
            
            # Plot chromagram
            librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', ax=axes[3])
            axes[3].set_title(f'Chromagram (Key: {self.features["key"]})')
            
            plt.tight_layout()
            save_path.parent.mkdir(exist_ok=True, parents=True)
            plt.savefig(save_path)
            plt.close()
            
            logger.info(f"Visualizations saved to {save_path}")
        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")
            logger.error(traceback.format_exc())


def analyze_directory(input_dir: str) -> Dict[str, Dict[str, Any]]:
    """
    Analyze all audio files in a directory and return a dictionary of results
    
    Args:
        input_dir: Directory containing audio files
        
    Returns:
        Dictionary mapping file names to feature dictionaries
    """
    input_path = Path(input_dir)
    logger.info(f"Analyzing audio files in {input_path}")
    
    # Find all audio files
    audio_extensions = ['.mp3', '.wav', '.flac', '.ogg', '.m4a']
    audio_files = []
    
    for ext in audio_extensions:
        audio_files.extend(list(input_path.glob(f"*{ext}")))
    
    if not audio_files:
        logger.warning(f"No audio files found in {input_path}")
        return {}
    
    logger.info(f"Found {len(audio_files)} audio files")
    
    # Analyze each file
    results = {}
    for file_path in tqdm(audio_files, desc="Analyzing audio files"):
        try:
            analyzer = AudioAnalyzer(file_path)
            features = analyzer.extract_all_features()
            analyzer.visualize_features()
            results[file_path.stem] = features
            logger.info(f"Successfully analyzed {file_path.name}")
        except Exception as e:
            logger.error(f"Error analyzing {file_path.name}: {str(e)}")
            logger.error(traceback.format_exc())
    
    logger.info(f"Analysis complete for {len(results)} files")
    return results


if __name__ == "__main__":
    # Test with a single file if run directly
    import sys
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        analyzer = AudioAnalyzer(file_path)
        features = analyzer.extract_all_features()
        analyzer.visualize_features()
        print(f"BPM: {features['bpm']:.1f}")
        print(f"Key: {features['key']}")
        print(f"Duration: {features['duration']:.2f}s")
    else:
        print("Please provide an audio file path.") 