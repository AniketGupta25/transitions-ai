import os
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
import traceback
from pydub import AudioSegment
import librosa
import soundfile as sf
import tempfile
import math

from transitions_ai.src.config import (
    SAMPLE_RATE, OUTPUT_FORMAT, OUTPUT_BITRATE,
    TRANSITION_EQ_GAIN, FILTER_FREQUENCIES, REVERB_AMOUNT,
    VOLUME_NORMALIZATION_TARGET, TEMP_DIR, OUTPUT_DIR
)
from transitions_ai.src.logger import get_logger

logger = get_logger("renderer")

class AudioRenderer:
    """
    Renders mashups by applying appropriate transitions and effects
    """
    
    def __init__(self, tracks_data: Dict[str, Dict[str, Any]], temp_dir: Optional[Path] = None):
        """
        Initialize the audio renderer
        
        Args:
            tracks_data: Dictionary mapping track names to their features and segments
            temp_dir: Directory for temporary files
        """
        self.tracks_data = tracks_data
        self.temp_dir = temp_dir or TEMP_DIR
        self.audio_cache = {}  # Cache for loaded audio files
        
        # Create temp directory if it doesn't exist
        self.temp_dir.mkdir(exist_ok=True, parents=True)
        
        logger.info(f"Initializing audio renderer with {len(tracks_data)} tracks")
    
    def render_mashup(self, segments: List[Dict[str, Any]], transitions: List[Dict[str, Any]], 
                     output_path: Optional[Path] = None) -> Path:
        """
        Render a complete mashup from segment and transition data
        
        Args:
            segments: List of segment data
            transitions: List of transition data
            output_path: Path to save the output file
            
        Returns:
            Path to the output file
        """
        if not segments:
            logger.error("No segments provided for rendering")
            raise ValueError("No segments provided for rendering")
            
        if len(segments) < 2:
            logger.warning("Only one segment provided, no transitions to render")
        
        # Set default output path if not provided
        if output_path is None:
            # Generate a name based on the tracks used
            tracks_used = sorted(set(segment['track'] for segment in segments))
            mashup_name = f"mashup_{'_'.join(tracks_used[:3])}"
            if len(tracks_used) > 3:
                mashup_name += f"_and_{len(tracks_used)-3}_more"
            
            output_path = OUTPUT_DIR / f"{mashup_name}.{OUTPUT_FORMAT}"
        
        logger.info(f"Rendering mashup to {output_path}")
        
        try:
            # Create an empty mashup
            mashup = None
            
            # Process each segment and transition
            for i, segment in enumerate(segments):
                try:
                    # Get segment audio
                    segment_audio = self._extract_segment_audio(
                        segment['track'], 
                        segment['start'], 
                        segment['end']
                    )
                    
                    # Apply any segment-specific processing
                    segment_audio = self._process_segment(segment_audio, segment)
                    
                    # Add the segment to the mashup
                    if mashup is None:
                        mashup = segment_audio
                    else:
                        # Get transition if not the first segment
                        transition = transitions[i-1] if i-1 < len(transitions) else {
                            'type': 'crossfade',
                            'duration': 2.0
                        }
                        
                        # Apply transition
                        try:
                            mashup = self._apply_transition(
                                mashup, 
                                segment_audio, 
                                transition
                            )
                        except Exception as transition_error:
                            logger.error(f"Error applying transition: {str(transition_error)}")
                            # Fallback to simple append
                            logger.warning("Falling back to simple append without transition")
                            mashup = mashup + segment_audio
                            
                except Exception as segment_error:
                    logger.error(f"Error processing segment {i}: {str(segment_error)}")
                    logger.error(traceback.format_exc())
                    # Continue with next segment
                    continue
            
            # Apply final processing to the entire mashup
            if mashup:
                try:
                    mashup = self._apply_final_processing(mashup)
                except Exception as process_error:
                    logger.error(f"Error in final processing: {str(process_error)}")
                    # Continue with unprocessed mashup
                
                # Export the mashup
                try:
                    output_path.parent.mkdir(exist_ok=True, parents=True)
                    
                    # Try different export formats if the first fails
                    export_formats = [OUTPUT_FORMAT]
                    if OUTPUT_FORMAT != 'mp3':
                        export_formats.append('mp3')
                    if OUTPUT_FORMAT != 'wav':
                        export_formats.append('wav')
                    
                    export_success = False
                    export_error = None
                    
                    for format_attempt in export_formats:
                        try:
                            actual_output_path = output_path
                            if format_attempt != OUTPUT_FORMAT:
                                actual_output_path = output_path.with_suffix(f".{format_attempt}")
                                
                            mashup.export(
                                actual_output_path, 
                                format=format_attempt, 
                                bitrate=OUTPUT_BITRATE
                            )
                            
                            logger.info(f"Mashup exported to {actual_output_path}")
                            export_success = True
                            return actual_output_path
                        except Exception as e:
                            export_error = e
                            logger.warning(f"Failed to export as {format_attempt}: {str(e)}")
                    
                    if not export_success:
                        logger.error(f"All export attempts failed: {str(export_error)}")
                        raise RuntimeError(f"Failed to export mashup: {str(export_error)}")
                        
                except Exception as export_error:
                    logger.error(f"Error exporting mashup: {str(export_error)}")
                    raise
            else:
                logger.error("Failed to create mashup - No audio generated")
                raise RuntimeError("Failed to create mashup - No audio generated")
                
        except Exception as e:
            logger.error(f"Error rendering mashup: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def _extract_segment_audio(self, track_name: str, start_time: float, end_time: float) -> AudioSegment:
        """
        Extract audio for a segment from a track
        
        Args:
            track_name: Name of the track
            start_time: Start time in seconds
            end_time: End time in seconds
            
        Returns:
            Audio segment
        """
        try:
            # Get track file path
            track_path = None
            for track_name_key, track_data in self.tracks_data.items():
                if track_name_key == track_name or (track_data.get('file_path') and Path(track_data['file_path']).stem == track_name):
                    track_path = track_data.get('file_path')
                    break
            
            if not track_path:
                logger.error(f"Track file path not found for {track_name}")
                # Create a silent segment as fallback
                logger.warning(f"Creating silent fallback for {track_name}")
                return AudioSegment.silent(duration=int((end_time - start_time) * 1000))
            
            # Convert to milliseconds for pydub
            start_ms = int(start_time * 1000)
            end_ms = int(end_time * 1000)
            
            # Check if we have the track in cache
            if track_name not in self.audio_cache:
                # Load the audio file
                logger.info(f"Loading audio for {track_name}")
                try:
                    self.audio_cache[track_name] = AudioSegment.from_file(track_path)
                except Exception as load_error:
                    logger.error(f"Failed to load audio for {track_name}: {str(load_error)}")
                    # Try different formats as a fallback
                    for format_attempt in ['mp3', 'wav', 'ogg', 'm4a']:
                        try:
                            logger.info(f"Attempting to load as {format_attempt}")
                            self.audio_cache[track_name] = AudioSegment.from_file(track_path, format=format_attempt)
                            break
                        except Exception:
                            pass
                    
                    # If still not loaded, create silent audio
                    if track_name not in self.audio_cache:
                        logger.warning(f"All load attempts failed, creating silent audio for {track_name}")
                        duration = self.tracks_data.get(track_name, {}).get('duration', 180) * 1000  # default 3 min
                        self.audio_cache[track_name] = AudioSegment.silent(duration=int(duration))
            
            # Extract the segment
            track_audio = self.audio_cache[track_name]
            
            # Ensure start and end times are within track bounds
            if start_ms >= track_audio.duration_seconds * 1000:
                logger.warning(f"Start time {start_time}s is beyond track duration {track_audio.duration_seconds}s")
                start_ms = 0
            
            if end_ms > track_audio.duration_seconds * 1000:
                logger.warning(f"End time {end_time}s is beyond track duration {track_audio.duration_seconds}s")
                end_ms = int(track_audio.duration_seconds * 1000)
                
            if start_ms >= end_ms:
                logger.warning(f"Invalid segment bounds: start={start_ms}ms, end={end_ms}ms")
                # Create a short segment from the beginning of the track
                segment_audio = track_audio[:min(10000, int(track_audio.duration_seconds * 1000))]
            else:
                segment_audio = track_audio[start_ms:end_ms]
            
            logger.debug(f"Extracted segment from {track_name}: {start_time:.2f}s - {end_time:.2f}s ({segment_audio.duration_seconds:.2f}s)")
            return segment_audio
            
        except Exception as e:
            logger.error(f"Error extracting segment audio: {str(e)}")
            logger.error(traceback.format_exc())
            # Return silent audio as fallback
            logger.warning(f"Returning silent audio due to extraction error for {track_name}")
            return AudioSegment.silent(duration=int((end_time - start_time) * 1000))
    
    def _process_segment(self, segment_audio: AudioSegment, segment_data: Dict[str, Any]) -> AudioSegment:
        """
        Apply processing to a segment
        
        Args:
            segment_audio: Audio segment
            segment_data: Segment metadata
            
        Returns:
            Processed audio segment
        """
        # Apply processing based on segment type
        segment_type = segment_data.get('type', 'unknown')
        
        # Example processing based on segment type
        if segment_type == 'intro':
            # Fade in
            segment_audio = segment_audio.fade_in(int(min(2000, segment_audio.duration_seconds * 1000 / 4)))
        elif segment_type == 'outro':
            # Fade out
            segment_audio = segment_audio.fade_out(int(min(3000, segment_audio.duration_seconds * 1000 / 3)))
        
        return segment_audio
    
    def _apply_transition(self, source_audio: AudioSegment, target_audio: AudioSegment, 
                         transition: Dict[str, Any]) -> AudioSegment:
        """
        Apply a transition between two audio segments using professional DJ techniques
        """
        transition_type = transition['type']
        transition_duration_ms = int(transition['duration'] * 1000)
        
        # Get transition BPM and musical info
        bpm = transition.get('bpm', 120)
        beats_per_second = bpm / 60
        beat_duration_ms = int(60000 / bpm)
        
        # Always align transitions to musical phrases (8 or 16 beats typically)
        phrase_duration_ms = beat_duration_ms * 8  # 8 beats = 2 bars
        phrases = max(2, round(transition_duration_ms / phrase_duration_ms))  # At least 2 phrases (16 beats)
        transition_duration_ms = phrase_duration_ms * phrases
        logger.info(f"Aligned transition to {phrases} phrases ({transition_duration_ms}ms at {bpm} BPM)")
        
        # Get energy levels and track info
        from_energy = transition.get('from_energy', 0.5)
        to_energy = transition.get('to_energy', 0.5)
        energy_increasing = to_energy >= from_energy
        
        from_segment_id = transition.get('from_segment', '')
        to_segment_id = transition.get('to_segment', '')
        from_track = from_segment_id.split('_')[0] if '_' in from_segment_id else 'unknown'
        to_track = to_segment_id.split('_')[0] if '_' in to_segment_id else 'unknown'
        
        logger.info(f"Creating {transition_type} transition from {from_track} to {to_track}")
        
        # Professional DJ transition techniques
        if transition_type == 'beatmatch_crossfade':
            # Professional beatmatched transition with EQ mixing
            logger.info("Using professional beatmatched transition")
            
            # Extract longer sections for more natural transitions
            overlap_ms = transition_duration_ms
            source_end = source_audio[-overlap_ms:]
            target_start = target_audio[:overlap_ms]
            
            # Split into frequency bands
            source_bands = self._split_frequency_bands(source_end)
            target_bands = self._split_frequency_bands(target_start)
            
            # Create DJ-style transition:
            # 1. First bring in the highs from the new track
            # 2. Then bring in the mids while reducing the old track's mids
            # 3. Finally swap the bass at a key moment
            
            # Calculate timing for each stage
            high_start_pos = 0.2  # Start bringing in highs at 20%
            mid_start_pos = 0.4   # Start mids transition at 40%
            bass_swap_pos = 0.8   # Swap bass at 80%
            
            # Initialize processed bands
            processed_bands = {
                'low': AudioSegment.silent(duration=overlap_ms),
                'mid': AudioSegment.silent(duration=overlap_ms),
                'high': AudioSegment.silent(duration=overlap_ms)
            }
            
            # Process in small chunks for smoother transitions
            chunk_size = beat_duration_ms  # Process one beat at a time
            num_chunks = overlap_ms // chunk_size
            
            for i in range(num_chunks):
                position = i / num_chunks
                chunk_start = i * chunk_size
                chunk_end = (i + 1) * chunk_size
                
                # Process each frequency band
                # High frequencies
                if position >= high_start_pos:
                    high_blend = min(1.0, (position - high_start_pos) / (1 - high_start_pos))
                    source_high = source_bands['high'][chunk_start:chunk_end]
                    target_high = target_bands['high'][chunk_start:chunk_end]
                    
                    # Crossfade highs
                    source_high = source_high.apply_gain((high_blend - 1) * 12)  # -12dB at full blend
                    target_high = target_high.apply_gain(-((1 - high_blend) * 12))
                    processed_bands['high'] = processed_bands['high'].overlay(
                        source_high.overlay(target_high), 
                        position=chunk_start
                    )
                else:
                    # Keep source highs
                    processed_bands['high'] = processed_bands['high'].overlay(
                        source_bands['high'][chunk_start:chunk_end],
                        position=chunk_start
                    )
                
                # Mid frequencies
                if position >= mid_start_pos:
                    mid_blend = min(1.0, (position - mid_start_pos) / (1 - mid_start_pos))
                    source_mid = source_bands['mid'][chunk_start:chunk_end]
                    target_mid = target_bands['mid'][chunk_start:chunk_end]
                    
                    # Crossfade mids
                    source_mid = source_mid.apply_gain((mid_blend - 1) * 9)  # -9dB at full blend
                    target_mid = target_mid.apply_gain(-((1 - mid_blend) * 9))
                    processed_bands['mid'] = processed_bands['mid'].overlay(
                        source_mid.overlay(target_mid),
                        position=chunk_start
                    )
                else:
                    # Keep source mids
                    processed_bands['mid'] = processed_bands['mid'].overlay(
                        source_bands['mid'][chunk_start:chunk_end],
                        position=chunk_start
                    )
                
                # Bass frequencies - quick swap at the transition point
                if position >= bass_swap_pos:
                    # Use target bass
                    processed_bands['low'] = processed_bands['low'].overlay(
                        target_bands['low'][chunk_start:chunk_end],
                        position=chunk_start
                    )
                else:
                    # Use source bass
                    processed_bands['low'] = processed_bands['low'].overlay(
                        source_bands['low'][chunk_start:chunk_end],
                        position=chunk_start
                    )
            
            # Combine the processed bands
            transition_audio = self._combine_frequency_bands(list(processed_bands.values()))
            
            # Create final transition
            source_without_end = source_audio[:-overlap_ms]
            target_remaining = target_audio[overlap_ms:]
            
            return source_without_end + transition_audio + target_remaining
            
        elif transition_type == 'filter_sweep':
            # Professional filter sweep transition
            logger.info("Using professional filter sweep transition")
            
            # Extract sections for transition
            overlap_ms = transition_duration_ms
            source_end = source_audio[-overlap_ms:]
            target_start = target_audio[:overlap_ms]
            
            # Create DJ-style filter sweep:
            # 1. Apply resonant filter sweep on the outgoing track
            # 2. Bring in the new track with complementary filtering
            # 3. Use beat-aligned filter movements
            
            chunk_size = beat_duration_ms
            num_chunks = overlap_ms // chunk_size
            
            # Initialize processed audio
            processed_source = AudioSegment.silent(duration=overlap_ms)
            processed_target = AudioSegment.silent(duration=overlap_ms)
            
            # Define filter sweep parameters
            if energy_increasing:
                # Building energy: sweep from low to high
                source_freq_start = 4000  # Start with everything
                source_freq_end = 200    # End with just bass
                target_freq_start = 200   # Start with just bass
                target_freq_end = 4000   # End with everything
            else:
                # Reducing energy: sweep from high to low
                source_freq_start = 200   # Start with everything
                source_freq_end = 4000    # End with just highs
                target_freq_start = 4000  # Start with just highs
                target_freq_end = 200     # End with everything
            
            for i in range(num_chunks):
                position = i / num_chunks
                chunk_start = i * chunk_size
                chunk_end = (i + 1) * chunk_size
                
                # Calculate filter frequencies with exponential curve for more musical movement
                curve_pos = 0.5 - 0.5 * math.cos(position * math.pi)  # Smooth cosine curve
                source_freq = source_freq_start + (source_freq_end - source_freq_start) * curve_pos
                target_freq = target_freq_start + (target_freq_end - target_freq_start) * curve_pos
                
                # Extract and process chunks
                source_chunk = source_end[chunk_start:chunk_end]
                target_chunk = target_start[chunk_start:chunk_end]
                
                if energy_increasing:
                    # Process source (filtering out highs gradually)
                    source_chunk = source_chunk.low_pass_filter(source_freq)
                    # Process target (bringing in highs gradually)
                    target_chunk = target_chunk.high_pass_filter(target_freq)
                else:
                    # Process source (filtering out lows gradually)
                    source_chunk = source_chunk.high_pass_filter(source_freq)
                    # Process target (bringing in lows gradually)
                    target_chunk = target_chunk.low_pass_filter(target_freq)
                
                # Apply volume curves
                source_vol = 1 - curve_pos
                target_vol = curve_pos
                source_chunk = source_chunk.apply_gain(-(1 - source_vol) * 3)  # Max -3dB reduction
                target_chunk = target_chunk.apply_gain(-(1 - target_vol) * 3)  # Max -3dB reduction
                
                # Add to processed audio
                processed_source = processed_source.overlay(source_chunk, position=chunk_start)
                processed_target = processed_target.overlay(target_chunk, position=chunk_start)
            
            # Combine processed audio
            transition_audio = processed_source.overlay(processed_target)
            
            # Create final transition
            source_without_end = source_audio[:-overlap_ms]
            target_remaining = target_audio[overlap_ms:]
            
            return source_without_end + transition_audio + target_remaining
            
        elif transition_type == 'harmonic_crossfade':
            # Professional harmonic transition
            logger.info("Using professional harmonic transition")
            
            # Use musical phrase length for harmonic transitions
            overlap_ms = transition_duration_ms
            source_end = source_audio[-overlap_ms:]
            target_start = target_audio[:overlap_ms]
            
            # Split into frequency bands
            source_bands = self._split_frequency_bands(source_end)
            target_bands = self._split_frequency_bands(target_start)
            
            # Process each band with musical timing
            chunk_size = beat_duration_ms
            num_chunks = overlap_ms // chunk_size
            
            processed_bands = {
                'low': AudioSegment.silent(duration=overlap_ms),
                'mid': AudioSegment.silent(duration=overlap_ms),
                'high': AudioSegment.silent(duration=overlap_ms)
            }
            
            # Define blend points for each frequency band
            blend_points = {
                'high': 0.3,  # Start high transition early
                'mid': 0.4,   # Mids follow shortly after
                'low': 0.7    # Bass transitions later
            }
            
            for i in range(num_chunks):
                position = i / num_chunks
                chunk_start = i * chunk_size
                chunk_end = (i + 1) * chunk_size
                
                # Process each band
                for band_name, blend_start in blend_points.items():
                    source_chunk = source_bands[band_name][chunk_start:chunk_end]
                    target_chunk = target_bands[band_name][chunk_start:chunk_end]
                    
                    if position >= blend_start:
                        # Calculate blend amount
                        blend = min(1.0, (position - blend_start) / (1 - blend_start))
                        
                        # Apply smooth power curve
                        curve_pos = math.pow(blend, 2)  # Quadratic curve for smoother blend
                        
                        # Apply volume curves with minimal reduction
                        source_chunk = source_chunk.apply_gain(-(curve_pos * 4))  # Max -4dB reduction
                        target_chunk = target_chunk.apply_gain(-((1 - curve_pos) * 4))
                        
                        # Combine chunks
                        mixed_chunk = source_chunk.overlay(target_chunk)
                    else:
                        # Keep source audio before blend point
                        mixed_chunk = source_chunk
                    
                    # Add to processed band
                    processed_bands[band_name] = processed_bands[band_name].overlay(
                        mixed_chunk,
                        position=chunk_start
                    )
            
            # Combine all bands
            transition_audio = self._combine_frequency_bands(list(processed_bands.values()))
            
            # Create final transition
            source_without_end = source_audio[:-overlap_ms]
            target_remaining = target_audio[overlap_ms:]
            
            return source_without_end + transition_audio + target_remaining
            
        else:
            # Professional crossfade
            logger.info("Using professional crossfade transition")
            
            overlap_ms = transition_duration_ms
            source_end = source_audio[-overlap_ms:]
            target_start = target_audio[:overlap_ms]
            
            # Process in beat-aligned chunks for more musical transitions
            chunk_size = beat_duration_ms
            num_chunks = overlap_ms // chunk_size
            
            processed_audio = AudioSegment.silent(duration=overlap_ms)
            
            for i in range(num_chunks):
                position = i / num_chunks
                chunk_start = i * chunk_size
                chunk_end = (i + 1) * chunk_size
                
                # Get audio chunks
                source_chunk = source_end[chunk_start:chunk_end]
                target_chunk = target_start[chunk_start:chunk_end]
                
                # Calculate blend using smooth curve
                curve_pos = 0.5 - 0.5 * math.cos(position * math.pi)
                
                # Apply minimal volume reduction
                source_chunk = source_chunk.apply_gain(-(curve_pos * 3))  # Max -3dB reduction
                target_chunk = target_chunk.apply_gain(-((1 - curve_pos) * 3))
                
                # Combine chunks
                mixed_chunk = source_chunk.overlay(target_chunk)
                
                # Add to processed audio
                processed_audio = processed_audio.overlay(mixed_chunk, position=chunk_start)
            
            # Create final transition
            source_without_end = source_audio[:-overlap_ms]
            target_remaining = target_audio[overlap_ms:]
            
            return source_without_end + processed_audio + target_remaining
    
    def _split_frequency_bands(self, audio: AudioSegment) -> Dict[str, AudioSegment]:
        """Split audio into frequency bands for professional mixing"""
        # Create frequency bands using available pydub filters
        low = audio.low_pass_filter(250)
        
        # Create mid band using combination of high and low pass
        mid_low = audio.high_pass_filter(250)
        mid = mid_low.low_pass_filter(2500)
        
        high = audio.high_pass_filter(2500)
        
        # Minimal volume adjustments for cleaner mixing
        low = low - 1.5  # -1.5 dB for cleaner bass
        mid = mid + 0.5  # +0.5 dB for presence
        high = high - 1  # -1 dB to prevent harshness
        
        return {
            'low': low,
            'mid': mid,
            'high': high
        }
    
    def _combine_frequency_bands(self, bands: List[AudioSegment]) -> AudioSegment:
        """Combine frequency bands back together with professional mixing techniques"""
        if not bands:
            return AudioSegment.empty()
            
        # Start with the mid frequencies as the base
        if len(bands) >= 2:
            result = bands[1]  # mid band
            # Add lows
            result = result.overlay(bands[0])  # low band
            # Add highs if available
            if len(bands) >= 3:
                result = result.overlay(bands[2])  # high band
        else:
            result = bands[0]
            
        return result
    
    def _apply_final_processing(self, audio: AudioSegment) -> AudioSegment:
        """
        Apply final processing to the entire mashup
        
        Args:
            audio: The complete mashup audio
            
        Returns:
            Processed audio
        """
        logger.info("Applying final processing to mashup")
        
        # Normalize volume
        target_dBFS = VOLUME_NORMALIZATION_TARGET
        change_in_dBFS = target_dBFS - audio.dBFS
        normalized_audio = audio.apply_gain(change_in_dBFS)
        
        # Add fade in/out to the entire mashup
        fade_duration_ms = min(3000, audio.duration_seconds * 1000 / 20)  # 5% of duration or 3 seconds max
        final_audio = normalized_audio.fade_in(int(fade_duration_ms)).fade_out(int(fade_duration_ms))
        
        return final_audio


def render_mashup(tracks_data: Dict[str, Dict[str, Any]], 
                segments: List[Dict[str, Any]], 
                transitions: List[Dict[str, Any]],
                output_path: Optional[Path] = None) -> Path:
    """
    Render a mashup from segments and transitions
    
    Args:
        tracks_data: Dictionary mapping track names to their features and segments
        segments: List of segment data
        transitions: List of transition data
        output_path: Path to save the output file
        
    Returns:
        Path to the output file
    """
    renderer = AudioRenderer(tracks_data)
    return renderer.render_mashup(segments, transitions, output_path)


if __name__ == "__main__":
    # Test code
    import json
    import sys
    from pathlib import Path
    
    # Load sample data from files if available
    if len(sys.argv) > 1:
        mashup_file = Path(sys.argv[1])
        tracks_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else mashup_file.parent
        
        # Load mashup data
        try:
            with open(mashup_file, 'r') as f:
                mashup_data = json.load(f)
                
            segments = mashup_data.get('segments', [])
            transitions = mashup_data.get('transitions', [])
            
            if not segments:
                logger.error("No segments found in mashup data")
                sys.exit(1)
                
            # Load track data
            tracks_data = {}
            for file_path in tracks_dir.glob("*.json"):
                if file_path != mashup_file:  # Skip the mashup file
                    with open(file_path, 'r') as f:
                        track_data = json.load(f)
                        tracks_data[file_path.stem] = track_data
            
            # Add file paths to tracks_data
            for track_name in tracks_data:
                # Look for audio files with matching names
                audio_files = list(tracks_dir.glob(f"{track_name}.*"))
                if audio_files:
                    tracks_data[track_name]['file_path'] = str(audio_files[0])
            
            # Render the mashup
            output_path = render_mashup(tracks_data, segments, transitions)
            print(f"Mashup rendered to {output_path}")
            
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            traceback.print_exc()
            sys.exit(1)
    else:
        print("Please provide a mashup data file and optionally a tracks directory.") 