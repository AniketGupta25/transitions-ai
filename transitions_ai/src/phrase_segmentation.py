import os
import numpy as np
import librosa
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import traceback

from transitions_ai.src.config import (
    SAMPLE_RATE, HOP_LENGTH, N_FFT, VISUALIZATION_DIR, TEMP_DIR,
    MIN_PHRASE_LENGTH_SECONDS, MAX_PHRASE_LENGTH_SECONDS,
    ENERGY_THRESHOLD, ONSET_THRESHOLD, BOUNDARY_DETECTION_SENSITIVITY,
    SEGMENT_PADDING_SECONDS
)
from transitions_ai.src.logger import get_logger, log_segmentation

logger = get_logger("segmentation")

class PhraseSegmenter:
    """
    Class for segmenting audio tracks into distinct musical phrases
    such as intro, verse, chorus, bridge, etc.
    """
    
    def __init__(self, audio_features: Dict[str, Any], track_name: str):
        """
        Initialize the segmenter with audio features
        
        Args:
            audio_features: Dictionary of audio features from AudioAnalyzer
            track_name: Name of the track
        """
        self.features = audio_features
        self.track_name = track_name
        logger.info(f"Initializing segmenter for {track_name}")
        
        # Convert feature lists back to numpy arrays if needed
        if isinstance(self.features.get('beat_times', []), list):
            self.features['beat_times'] = np.array(self.features['beat_times'])
        
        if isinstance(self.features.get('onset_times', []), list):
            self.features['onset_times'] = np.array(self.features['onset_times'])
        
        if isinstance(self.features.get('rms_envelope', []), list):
            self.features['rms_envelope'] = np.array(self.features['rms_envelope'])
            
        if isinstance(self.features.get('chroma', []), list):
            self.features['chroma'] = np.array(self.features['chroma'])
        
        # Initialize results
        self.segments = []
        self.segment_cache_file = TEMP_DIR / f"{track_name}_segments.pkl"
    
    def segment_track(self) -> List[Dict[str, Any]]:
        """
        Segment the track into distinct musical phrases using multiple techniques
        
        Returns:
            List of segment dictionaries with start, end, type, and features
        """
        logger.info(f"Segmenting track: {self.track_name}")
        
        try:
            # Combine multiple segmentation techniques
            structural_segments = self.detect_structural_segments()
            energy_segments = self.segment_by_energy()
            harmony_segments = self.segment_by_harmony()
            
            # Combine and merge overlapping segments
            self.segments = self._merge_segments([
                structural_segments,
                energy_segments,
                harmony_segments
            ])
            
            # If we still don't have segments, create fallback segments
            if not self.segments:
                logger.warning(f"No segments found after merging - creating fallbacks")
                self.segments = self._create_fallback_segments()
            
            # Filter segments by minimum length
            min_length = MIN_PHRASE_LENGTH_SECONDS
            self.segments = [s for s in self.segments if s['duration'] >= min_length]
            
            # If filtering left us with no segments, create fallbacks
            if not self.segments:
                logger.warning(f"No segments left after filtering - creating fallbacks")
                self.segments = self._create_fallback_segments()
            
            # Classify segments by type
            self._classify_segments()
            
            # Sort segments by start time
            self.segments = sorted(self.segments, key=lambda s: s['start'])
            
            # Add unique IDs to segments
            for i, segment in enumerate(self.segments):
                segment['id'] = f"{self.track_name}_{i+1}"
            
            # Log results
            logger.info(f"Created {len(self.segments)} segments for {self.track_name}")
            
            return self.segments
        except Exception as e:
            logger.error(f"Error segmenting track: {str(e)}")
            logger.error(traceback.format_exc())
            # Return fallback segments if segmentation fails
            self.segments = self._create_fallback_segments()
            
            # Add unique IDs
            for i, segment in enumerate(self.segments):
                segment['id'] = f"{self.track_name}_{i+1}"
                
            logger.info(f"Created {len(self.segments)} fallback segments after error")
            return self.segments
    
    def detect_structural_segments(self) -> List[Dict[str, Any]]:
        """
        Detect structural segments based on repetition and similarity
        
        Returns:
            List of segment dictionaries
        """
        logger.info(f"Detecting structural segments for {self.track_name}")
        
        try:
            # Get beat-synchronized chroma features
            sr = SAMPLE_RATE
            hop_length = HOP_LENGTH
            
            # Get chroma features
            chroma = self.features['chroma']
            
            # Convert to beat-synchronized chroma
            beat_times = self.features['beat_times']
            beat_frames = librosa.time_to_frames(beat_times, sr=sr, hop_length=hop_length)
            beat_chroma = librosa.util.sync(chroma, beat_frames, aggregate=np.median)
            
            # Compute self-similarity matrix
            S = librosa.segment.recurrence_matrix(beat_chroma, mode='affinity', width=3)
            
            # Enhance diagonals with a median filter
            S = librosa.segment.path_enhance(S, 15)
            
            # Detect segments using spectral clustering
            boundary_frames = librosa.segment.agglomerative(S, 6)
            boundary_times = librosa.frames_to_time(librosa.util.fix_frames(boundary_frames, 
                                                    x_min=0, 
                                                    x_max=beat_chroma.shape[1]), 
                                                    sr=sr, 
                                                    hop_length=hop_length)
            
            # Create segment list
            segments = []
            for i in range(len(boundary_times) - 1):
                start = boundary_times[i]
                end = boundary_times[i + 1]
                
                segment = {
                    'start': float(start),
                    'end': float(end),
                    'duration': float(end - start),
                    'type': 'unknown',
                    'source': 'structural'
                }
                segments.append(segment)
            
            logger.info(f"Found {len(segments)} structural segments")
            return segments
        except Exception as e:
            logger.error(f"Error detecting structural segments: {str(e)}")
            logger.error(traceback.format_exc())
            return []
    
    def segment_by_energy(self) -> List[Dict[str, Any]]:
        """
        Segment track based on energy levels (quiet vs loud sections)
        
        Returns:
            List of segment dictionaries
        """
        logger.info(f"Segmenting by energy for {self.track_name}")
        
        try:
            # Get RMS energy
            if 'rms_envelope' not in self.features:
                logger.warning("RMS envelope not found in features")
                return []
            
            rms = self.features['rms_envelope']
            hop_length = HOP_LENGTH
            sr = SAMPLE_RATE
            
            # Normalize RMS
            rms_norm = rms / np.max(rms)
            
            # Smooth the energy curve
            window_size = int(1.0 * sr / hop_length)  # 1 second window
            rms_smooth = np.convolve(rms_norm, np.ones(window_size)/window_size, mode='same')
            
            # Find significant changes in energy
            threshold = ENERGY_THRESHOLD
            energy_diff = np.abs(np.diff(rms_smooth))
            peaks = librosa.util.peak_pick(energy_diff, 
                                          pre_max=10, 
                                          post_max=10, 
                                          pre_avg=10, 
                                          post_avg=10, 
                                          delta=threshold, 
                                          wait=20)
            
            # Convert peaks to times
            boundary_times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop_length)
            
            # Add start and end points
            boundary_times = np.concatenate([[0], boundary_times, [self.features['duration']]])
            
            # Create segments
            segments = []
            for i in range(len(boundary_times) - 1):
                start = boundary_times[i]
                end = boundary_times[i + 1]
                
                # Calculate average energy in this segment
                start_frame = librosa.time_to_frames(start, sr=sr, hop_length=hop_length)
                end_frame = librosa.time_to_frames(end, sr=sr, hop_length=hop_length)
                
                # Ensure frames are within bounds
                start_frame = max(0, start_frame)
                end_frame = min(len(rms_norm) - 1, end_frame)
                
                # Calculate average energy
                avg_energy = np.mean(rms_norm[start_frame:end_frame+1]) if end_frame > start_frame else 0
                
                segment = {
                    'start': float(start),
                    'end': float(end),
                    'duration': float(end - start),
                    'type': 'unknown',
                    'avg_energy': float(avg_energy),
                    'source': 'energy'
                }
                segments.append(segment)
            
            logger.info(f"Found {len(segments)} energy-based segments")
            return segments
        except Exception as e:
            logger.error(f"Error in energy-based segmentation: {str(e)}")
            logger.error(traceback.format_exc())
            return []
    
    def segment_by_harmony(self) -> List[Dict[str, Any]]:
        """
        Segment track based on harmonic changes
        
        Returns:
            List of segment dictionaries
        """
        logger.info(f"Segmenting by harmony for {self.track_name}")
        
        try:
            # Get chroma features
            if 'chroma' not in self.features:
                logger.warning("Chroma features not found")
                return []
            
            chroma = self.features['chroma']
            sr = SAMPLE_RATE
            hop_length = HOP_LENGTH
            
            # Detect harmonic changes
            harmonic_changes = np.sum(np.abs(np.diff(chroma, axis=1)), axis=0)
            harmonic_changes = np.concatenate([[0], harmonic_changes])
            
            # Smooth the changes
            window_size = int(0.5 * sr / hop_length)  # 0.5 second window
            harmonic_changes_smooth = np.convolve(harmonic_changes, 
                                                 np.ones(window_size)/window_size, 
                                                 mode='same')
            
            # Find peaks in harmonic changes
            peaks = librosa.util.peak_pick(harmonic_changes_smooth, 
                                          pre_max=20, 
                                          post_max=20, 
                                          pre_avg=20, 
                                          post_avg=20, 
                                          delta=0.03, 
                                          wait=20)
            
            # Convert peaks to times
            boundary_times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop_length)
            
            # Add start and end points
            boundary_times = np.concatenate([[0], boundary_times, [self.features['duration']]])
            
            # Create segments
            segments = []
            for i in range(len(boundary_times) - 1):
                start = boundary_times[i]
                end = boundary_times[i + 1]
                
                segment = {
                    'start': float(start),
                    'end': float(end),
                    'duration': float(end - start),
                    'type': 'unknown',
                    'source': 'harmony'
                }
                segments.append(segment)
            
            logger.info(f"Found {len(segments)} harmony-based segments")
            return segments
        except Exception as e:
            logger.error(f"Error in harmony-based segmentation: {str(e)}")
            logger.error(traceback.format_exc())
            return []
    
    def _merge_segments(self, segment_lists: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Merge multiple segment lists and resolve overlaps
        
        Args:
            segment_lists: List of segment lists from different techniques
            
        Returns:
            Merged list of segments
        """
        logger.info(f"Merging segments for {self.track_name}")
        
        # Flatten segment lists
        all_segments = []
        for segments in segment_lists:
            all_segments.extend(segments)
        
        if not all_segments:
            logger.warning("No segments found to merge - creating fallback segments")
            return self._create_fallback_segments()
        
        # Sort segments by start time
        all_segments = sorted(all_segments, key=lambda s: s['start'])
        
        # Try different segmentation strategies
        
        # First, try using structural segments as base
        structural_segments = [s for s in all_segments if s['source'] == 'structural']
        merged_segments = []
        
        # If we have enough structural segments, use them
        if structural_segments and len(structural_segments) >= 2:
            logger.info(f"Using {len(structural_segments)} structural segments as base")
            merged_segments = structural_segments
        else:
            # Try energy-based segmentation
            energy_segments = [s for s in all_segments if s['source'] == 'energy']
            min_length = MIN_PHRASE_LENGTH_SECONDS
            
            if energy_segments and len(energy_segments) >= 2:
                logger.info(f"Using {len(energy_segments)} energy-based segments")
                merged_segments = [s for s in energy_segments if s['duration'] >= min_length]
            
            # If that didn't work, try harmony-based segments
            if not merged_segments:
                harmony_segments = [s for s in all_segments if s['source'] == 'harmony']
                if harmony_segments:
                    logger.info(f"Found {len(harmony_segments)} harmony segments")
                    # Filter to reasonable-sized segments
                    decent_segments = [s for s in harmony_segments if min_length <= s['duration'] <= MAX_PHRASE_LENGTH_SECONDS]
                    
                    # Take every nth segment to get a reasonable number (aim for 4-8 segments)
                    if decent_segments:
                        step = max(1, len(decent_segments) // 6)  # aim for about 6 segments
                        merged_segments = decent_segments[::step]
                        logger.info(f"Selected {len(merged_segments)} harmony segments")
                
                # If we still don't have segments, merge adjacent segments
                if not merged_segments:
                    logger.info("Merging adjacent segments")
                    merged_segments = []
                    current = None
                    
                    for segment in all_segments:
                        if current is None:
                            current = segment.copy()
                        elif segment['start'] <= current['end'] + 1.0:
                            # Merge overlapping segments
                            current['end'] = max(current['end'], segment['end'])
                            current['duration'] = current['end'] - current['start']
                        else:
                            # Found a gap, finish current segment and start a new one
                            if current['duration'] >= min_length:
                                merged_segments.append(current)
                            current = segment.copy()
                    
                    # Add the last segment
                    if current and current['duration'] >= min_length:
                        merged_segments.append(current)
                    
                    logger.info(f"Created {len(merged_segments)} merged segments")
        
        # If we still don't have enough segments, create some
        if len(merged_segments) < 2:
            logger.warning("Not enough segments found - creating fallback segments")
            return self._create_fallback_segments()
        
        # Make sure segments have avg_energy
        for segment in merged_segments:
            if 'avg_energy' not in segment:
                # Calculate average energy for this segment
                start_time = segment['start']
                end_time = segment['end']
                start_frame = librosa.time_to_frames(start_time, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
                end_frame = librosa.time_to_frames(end_time, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
                
                # Ensure frames are within bounds
                start_frame = max(0, start_frame)
                end_frame = min(len(self.features['rms_envelope']) - 1, end_frame)
                
                # Calculate energy
                if start_frame < end_frame and 'rms_envelope' in self.features:
                    rms = self.features['rms_envelope']
                    avg_energy = float(np.mean(rms[start_frame:end_frame]) / np.max(rms))
                else:
                    avg_energy = 0.5
                
                segment['avg_energy'] = avg_energy
        
        logger.info(f"Merged into {len(merged_segments)} segments")
        return merged_segments
    
    def _create_fallback_segments(self) -> List[Dict[str, Any]]:
        """
        Create fallback segments when automatic segmentation fails
        
        Returns:
            List of segment dictionaries
        """
        # Determine reasonable number of segments based on track duration
        duration = self.features.get('duration', 0)
        if duration <= 0:
            logger.error("Invalid duration for track")
            # If we somehow have an invalid duration, set a default
            duration = 240  # 4 minutes as a fallback duration
        
        # Create approximately one segment per 30-45 seconds, with min of 4 and max of 10
        segment_duration = min(45, max(30, duration / 6))
        num_segments = max(4, min(10, int(duration / segment_duration)))
        segment_duration = duration / num_segments
        
        logger.info(f"Creating {num_segments} fallback segments of ~{segment_duration:.2f}s each")
        
        segments = []
        for i in range(num_segments):
            start = i * segment_duration
            end = min((i + 1) * segment_duration, duration)  # Ensure we don't exceed track duration
            
            # Determine segment type based on position
            if i == 0:
                segment_type = 'intro'
            elif i == num_segments - 1:
                segment_type = 'outro'
            elif i % 3 == 1:  # Every third segment (after intro) is a chorus
                segment_type = 'chorus'
            elif i % 3 == 2:  # Every third after chorus is a bridge
                segment_type = 'bridge'
            else:
                segment_type = 'verse'
            
            # Get BPM and key from features if available
            bpm = self.features.get('bpm', 120)
            key = self.features.get('key', 'C major')
            
            # Create segment
            segment = {
                'start': float(start),
                'end': float(end),
                'duration': float(end - start),
                'type': segment_type,
                'source': 'fallback',
                'avg_energy': 0.5,  # Default energy value
                'track': self.track_name,  # Ensure track name is included
                'bpm': bpm,
                'key': key
            }
            segments.append(segment)
        
        return segments
    
    def _classify_segments(self) -> None:
        """
        Classify segments by type (intro, verse, chorus, bridge, etc.)
        based on their features
        """
        logger.info(f"Classifying segments for {self.track_name}")
        
        if not self.segments:
            logger.warning("No segments to classify")
            return
            
        try:
            # Get features for each segment
            for segment in self.segments:
                # Extract segment time range
                start = segment['start']
                end = segment['end']
                
                # Get average energy
                avg_energy = segment.get('avg_energy', 0.5)
                
                # Check if this is the first or last segment
                is_first = segment == self.segments[0]
                is_last = segment == self.segments[-1]
                
                # Calculate relative position in the track
                rel_position = start / self.features['duration']
                
                # Calculate segment features
                
                # Get frames corresponding to the segment
                sr = SAMPLE_RATE
                hop_length = HOP_LENGTH
                start_frame = librosa.time_to_frames(start, sr=sr, hop_length=hop_length)
                end_frame = librosa.time_to_frames(end, sr=sr, hop_length=hop_length)
                
                # Ensure frames are within bounds
                start_frame = max(0, start_frame)
                end_frame = min(len(self.features['rms_envelope']) - 1, end_frame)
                
                # Calculate features if we have valid frames
                if end_frame > start_frame:
                    # Calculate spectral features if available
                    if isinstance(self.features.get('spectral_contrast', None), list):
                        segment['spectral_contrast'] = self.features['spectral_contrast']
                    
                    # Calculate segment complexity based on chroma variance
                    if isinstance(self.features.get('chroma', None), np.ndarray):
                        chroma_segment = self.features['chroma'][:, start_frame:end_frame]
                        segment['chroma_variance'] = float(np.mean(np.var(chroma_segment, axis=1)))
                
                # Store additional metadata
                segment['is_first'] = is_first
                segment['is_last'] = is_last
                segment['rel_position'] = float(rel_position)
            
            # Perform the classification based on heuristics and features
            
            # Use relative position, energy, and duration to classify
            for segment in self.segments:
                rel_pos = segment['rel_position']
                duration = segment['duration']
                energy = segment.get('avg_energy', 0.5)
                is_first = segment['is_first']
                is_last = segment['is_last']
                
                # Simple heuristic classification
                if is_first and rel_pos < 0.2:
                    segment_type = 'intro'
                elif is_last and rel_pos > 0.8:
                    segment_type = 'outro'
                elif energy > 0.7:
                    segment_type = 'chorus'
                elif energy < 0.4:
                    segment_type = 'bridge' if 0.4 < rel_pos < 0.8 else 'verse'
                else:
                    segment_type = 'verse'
                
                segment['type'] = segment_type
            
            # Try to add additional chorus identification using repetition pattern
            # (This is a simplification; real chorus detection requires more complex analysis)
            chorus_energy_threshold = 0.65
            high_energy_segments = [s for s in self.segments 
                                  if s.get('avg_energy', 0) > chorus_energy_threshold 
                                  and s['duration'] > MIN_PHRASE_LENGTH_SECONDS]
            
            if high_energy_segments:
                # The segment with highest energy is likely a chorus
                max_energy_segment = max(high_energy_segments, key=lambda s: s.get('avg_energy', 0))
                max_energy_segment['type'] = 'chorus'
                
                # Look for similar high-energy segments
                for segment in high_energy_segments:
                    if segment != max_energy_segment and abs(segment['duration'] - max_energy_segment['duration']) < 5:
                        segment['type'] = 'chorus'
            
            logger.info(f"Classified {len(self.segments)} segments")
        except Exception as e:
            logger.error(f"Error classifying segments: {str(e)}")
            logger.error(traceback.format_exc())
    
    def visualize_segmentation(self, save_path: Optional[Path] = None) -> None:
        """Generate visualization of the segmentation results"""
        if not save_path:
            save_path = VISUALIZATION_DIR / f"{self.track_name}_segmentation.png"
        
        logger.info(f"Visualizing segmentation for {self.track_name}")
        
        if not self.segments:
            logger.warning("No segments to visualize")
            return
            
        try:
            # Setup plot
            plt.figure(figsize=(12, 6))
            
            # Waveform or energy
            if isinstance(self.features.get('rms_envelope', None), np.ndarray) or isinstance(self.features.get('rms_envelope', None), list):
                rms = np.array(self.features['rms_envelope']) if isinstance(self.features.get('rms_envelope', None), list) else self.features['rms_envelope']
                times = librosa.times_like(rms, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
                plt.plot(times, rms / np.max(rms), color='gray', alpha=0.5)
            
            # Plot segment boundaries
            colors = {
                'intro': 'blue',
                'verse': 'green',
                'chorus': 'red',
                'bridge': 'purple',
                'outro': 'orange',
                'unknown': 'gray'
            }
            
            # Plot colored regions for each segment
            for segment in self.segments:
                start = segment['start']
                end = segment['end']
                segment_type = segment['type']
                color = colors.get(segment_type, 'gray')
                
                # Add colored background for the segment
                plt.axvspan(start, end, alpha=0.2, color=color)
                
                # Add segment label
                plt.text((start + end) / 2, 0.9, segment_type, 
                        horizontalalignment='center',
                        verticalalignment='center')
            
            # Add markers for segment boundaries
            for segment in self.segments:
                plt.axvline(x=segment['start'], color='black', linestyle='--', alpha=0.5)
            plt.axvline(x=self.segments[-1]['end'], color='black', linestyle='--', alpha=0.5)
            
            # Set labels and title
            plt.title(f"Segmentation: {self.track_name}")
            plt.xlabel("Time (s)")
            plt.ylabel("Energy / Segment Type")
            plt.ylim(0, 1)
            plt.grid(alpha=0.3)
            
            # Save figure
            save_path.parent.mkdir(exist_ok=True, parents=True)
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
            
            logger.info(f"Segmentation visualization saved to {save_path}")
        except Exception as e:
            logger.error(f"Error visualizing segmentation: {str(e)}")
            logger.error(traceback.format_exc())


def segment_track(audio_features: Dict[str, Any], track_name: str) -> List[Dict[str, Any]]:
    """
    Segment a track into musical phrases
    
    Args:
        audio_features: Dictionary of audio features from AudioAnalyzer
        track_name: Name of the track
        
    Returns:
        List of segment dictionaries
    """
    segmenter = PhraseSegmenter(audio_features, track_name)
    segments = segmenter.segment_track()
    
    # If no segments were detected, create fallback segments
    if not segments:
        logger.warning(f"No segments detected for {track_name}, creating fallbacks")
        segments = segmenter._create_fallback_segments()
    
    # Ensure each segment has a track field
    for segment in segments:
        if 'track' not in segment:
            segment['track'] = track_name
    
    # Visualize the segmentation
    segmenter.visualize_segmentation()
    
    # Log segment counts
    logger = get_logger("segmentation")
    logger.info(f"Segmentation completed for {track_name}")
    logger.info(f"Found {len(segments)} segments")
    
    return segments


if __name__ == "__main__":
    # Test with sample data if run directly
    import pickle
    import sys
    from transitions_ai.src.audio_analysis import AudioAnalyzer
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        analyzer = AudioAnalyzer(file_path)
        features = analyzer.extract_all_features()
        
        track_name = Path(file_path).stem
        segments = segment_track(features, track_name)
        
        print(f"Found {len(segments)} segments:")
        for i, segment in enumerate(segments):
            print(f"{i+1}. {segment['type']}: {segment['start']:.2f}s - {segment['end']:.2f}s (duration: {segment['duration']:.2f}s)")
    else:
        print("Please provide an audio file path.") 