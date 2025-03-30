import os
import sys
import time
from pathlib import Path
import json
import random
import traceback

from transitions_ai.src.config import (
    INPUT_DIR, OUTPUT_DIR, TEMP_DIR, MAX_MASHUP_LENGTH_MINUTES,
    MIN_SONGS_IN_MASHUP, MAX_CONSECUTIVE_PHRASES_FROM_SAME_SONG
)
from transitions_ai.src.logger import get_logger
from transitions_ai.src.transition_engine import build_transition_graph
from transitions_ai.src.mashup_generator import generate_mashup, MashupGenerator
from transitions_ai.src.audio_renderer import render_mashup

logger = get_logger("test_mashup")

def main():
    """Test script for generating a mashup from existing analysis data"""
    # Setup basic directories
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    data_dir = output_dir / "data"
    if not data_dir.exists():
        logger.error(f"Data directory {data_dir} does not exist. Run main.py first to analyze tracks.")
        return
    
    # Load tracks data from data directory
    tracks_data = {}
    logger.info(f"Loading track data from {data_dir}")
    
    for data_file in data_dir.glob("*.json"):
        try:
            with open(data_file, 'r') as f:
                track_data = json.load(f)
                track_name = data_file.stem
                
                # Verify this track has segments
                if 'segments' not in track_data or not track_data['segments']:
                    logger.warning(f"No segments found for track {track_name}, creating fallback segments")
                    # Create basic fallback segments
                    duration = track_data.get('duration', 240.0)
                    bpm = track_data.get('bpm', 120.0)
                    key = track_data.get('key', 'C major')
                    
                    num_segments = max(4, min(6, int(duration / 30)))
                    segment_duration = duration / num_segments
                    
                    segments = []
                    for i in range(num_segments):
                        start = i * segment_duration
                        end = min((i + 1) * segment_duration, duration)
                        segment_type = 'intro' if i == 0 else 'outro' if i == num_segments - 1 else 'verse' if i % 2 == 0 else 'chorus'
                        
                        segments.append({
                            'id': f"{track_name}_{i+1}",
                            'track': track_name,
                            'start': float(start),
                            'end': float(end),
                            'duration': float(end - start),
                            'type': segment_type,
                            'source': 'fallback',
                            'avg_energy': 0.5,
                            'bpm': bpm,
                            'key': key
                        })
                    
                    track_data['segments'] = segments
                
                # Make sure each segment has the track field
                for segment in track_data.get('segments', []):
                    if 'track' not in segment:
                        segment['track'] = track_name
                
                # Verify file path exists and update if necessary
                for ext in ['.wav', '.mp3', '.flac', '.ogg', '.m4a']:
                    potential_path = INPUT_DIR / f"{track_name}{ext}"
                    if potential_path.exists():
                        track_data['file_path'] = str(potential_path)
                        break
                
                tracks_data[track_name] = track_data
                logger.info(f"Loaded track {track_name} with {len(track_data.get('segments', []))} segments")
        except Exception as e:
            logger.error(f"Error loading {data_file}: {str(e)}")
    
    if not tracks_data:
        logger.error("No track data loaded. Cannot proceed.")
        return
    
    # Make sure we have at least 2 tracks for a proper mashup
    if len(tracks_data) < 2:
        logger.error(f"Need at least 2 tracks for a mashup, but only found {len(tracks_data)}.")
        return
    
    # Print information about loaded tracks
    logger.info(f"Loaded {len(tracks_data)} tracks:")
    for track_name, track_data in tracks_data.items():
        logger.info(f"  Track: {track_name}")
        logger.info(f"    Duration: {track_data.get('duration', 0):.2f}s")
        logger.info(f"    BPM: {track_data.get('bpm', 0):.1f}")
        logger.info(f"    Key: {track_data.get('key', 'unknown')}")
        segment_count = len(track_data.get('segments', []))
        logger.info(f"    Segments: {segment_count}")
        for i, segment in enumerate(track_data.get('segments', [])):
            logger.info(f"      {i+1}. {segment['type']}: {segment['start']:.1f}s-{segment['end']:.1f}s (id: {segment['id']})")
            if 'id' not in segment:
                logger.error(f"      MISSING ID in segment {i+1}")
    
    # Build transition graph with high randomness for maximum compatibility
    logger.info("Building transition graph with high randomness")
    transition_matrix = build_transition_graph(tracks_data, randomness_boost=2.0)
    
    # Print transition statistics
    transition_count = sum(len(v) for v in transition_matrix.values())
    logger.info(f"Built transition matrix with {transition_count} transitions across {len(transition_matrix)} segments")
    
    # Check which tracks have transitions to other tracks
    track_to_track = {}
    for from_id, transitions in transition_matrix.items():
        from_track = from_id.split('_')[0] if '_' in from_id else from_id
        if from_track not in track_to_track:
            track_to_track[from_track] = set()
        
        for to_id in transitions.keys():
            to_track = to_id.split('_')[0] if '_' in to_id else to_id
            if to_track != from_track:
                track_to_track[from_track].add(to_track)
    
    # Print track connectivity
    for track, connected_tracks in track_to_track.items():
        logger.info(f"Track {track} can transition to: {', '.join(connected_tracks) if connected_tracks else 'NONE'}")
    
    # Generate mashup with short duration for testing
    duration = 0.5  # Half a minute
    logger.info(f"Generating test mashup (target duration: {duration} minutes)")
    
    try:
        # Try the normal generation first
        try:
            mashup_segments, mashup_transitions = generate_mashup(
                tracks_data, 
                transition_matrix,
                target_duration=duration,
                min_tracks=2,
                max_consecutive=2,
                randomness_boost=2.0
            )
        except Exception as e:
            logger.error(f"Error in automatic mashup generation: {str(e)}")
            logger.error(traceback.format_exc())
            
            # If automatic generation fails, create a simple mashup manually
            logger.info("Falling back to simple mashup creation")
            generator = MashupGenerator(tracks_data, transition_matrix)
            mashup_segments, mashup_transitions = generator._create_simple_mashup(tracks_data, duration * 60)
        
        if not mashup_segments:
            logger.error("Failed to generate mashup - no segments returned")
            return
        
        logger.info(f"Generated mashup with {len(mashup_segments)} segments and {len(mashup_transitions)} transitions")
        
        # Check which tracks are included in the mashup
        tracks_in_mashup = set(segment['track'] for segment in mashup_segments)
        logger.info(f"Tracks in mashup: {', '.join(tracks_in_mashup)}")
        
        # Print mashup structure
        print("\nMashup Structure:")
        total_duration = 0
        for i, segment in enumerate(mashup_segments):
            print(f"  {i+1}. {segment['track']} - {segment['type']}: {segment['duration']:.2f}s")
            total_duration += segment['duration']
            
            # Add transition if not the last segment
            if i < len(mashup_transitions):
                transition = mashup_transitions[i]
                print(f"     ↓ [{transition['type']}] ({transition['duration']:.2f}s, score: {transition['score']:.2f})")
                total_duration += transition['duration']
        
        print(f"\nTotal mashup duration: {total_duration:.2f}s ({total_duration/60:.2f} minutes)")
        
        # Render the mashup
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_name = f"test_mashup_{timestamp}"
        output_path = output_dir / f"{output_name}.wav"
        
        logger.info(f"Rendering mashup to {output_path}")
        final_output_path = render_mashup(tracks_data, mashup_segments, mashup_transitions, output_path=output_path)
        logger.info(f"Mashup rendered to {final_output_path}")
        print(f"\nMashup saved to: {final_output_path}")
        
    except Exception as e:
        logger.error(f"Error generating mashup: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        logger.error(f"Unexpected error: {str(e)}", exc_info=True) 