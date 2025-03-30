import os
import sys
import time
from pathlib import Path
import json
import random

from transitions_ai.src.config import (
    INPUT_DIR, OUTPUT_DIR, TEMP_DIR
)
from transitions_ai.src.logger import get_logger
from transitions_ai.src.audio_renderer import render_mashup

logger = get_logger("simple_mashup")

def main():
    """
    Creates a simple mashup directly using segments from multiple tracks
    without relying on the transition engine or automatic mashup generation
    """
    # Setup directories
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    data_dir = output_dir / "data"
    if not data_dir.exists():
        logger.error(f"Data directory {data_dir} does not exist. Run main.py first to analyze tracks.")
        return
    
    # Load tracks data
    logger.info(f"Loading track data from {data_dir}")
    tracks_data = {}
    
    for data_file in data_dir.glob("*.json"):
        try:
            with open(data_file, 'r') as f:
                track_data = json.load(f)
                track_name = data_file.stem
                tracks_data[track_name] = track_data
                logger.info(f"Loaded track {track_name}")
        except Exception as e:
            logger.error(f"Error loading {data_file}: {str(e)}")
    
    # Make sure we have at least 2 tracks
    if len(tracks_data) < 2:
        logger.error(f"Need at least 2 tracks for a mashup, but only found {len(tracks_data)}.")
        return
    
    # Get track names
    track_names = list(tracks_data.keys())
    logger.info(f"Available tracks: {', '.join(track_names)}")
    
    # Select tracks for mashup (all available tracks)
    selected_tracks = track_names.copy()
    logger.info(f"Selected tracks for mashup: {', '.join(selected_tracks)}")
    
    # Create a manual mashup
    segments = []
    transitions = []
    
    # Create a track sequence with interesting structure, repeating tracks
    # Create pattern: A -> B -> C -> A -> B -> A -> C
    track_sequence = []
    if len(selected_tracks) >= 3:
        track_sequence = [
            selected_tracks[0],  # Start with track A
            selected_tracks[1],  # Switch to track B
            selected_tracks[2],  # Switch to track C
            selected_tracks[0],  # Back to track A
            selected_tracks[1],  # Back to track B
            selected_tracks[0],  # Finish with track A
            selected_tracks[2]   # Final track C
        ]
    elif len(selected_tracks) == 2:
        track_sequence = [
            selected_tracks[0],  # Start with track A
            selected_tracks[1],  # Switch to track B
            selected_tracks[0],  # Back to track A
            selected_tracks[1],  # Back to track B
            selected_tracks[0]   # Finish with track A
        ]
    
    # For each track in the sequence, pick appropriate segments
    current_segment_ids = {track: 0 for track in selected_tracks}
    
    for i, track_name in enumerate(track_sequence):
        track_data = tracks_data[track_name]
        
        # Ensure track has segments
        if 'segments' not in track_data or not track_data['segments']:
            logger.warning(f"No segments found for {track_name}, creating basic segments")
            duration = track_data.get('duration', 240.0)
            create_fallback_segments(track_data, track_name, duration)
        
        # Ensure file path is set
        if 'file_path' not in track_data:
            for ext in ['.wav', '.mp3', '.flac', '.ogg', '.m4a']:
                audio_file = Path(INPUT_DIR) / f"{track_name}{ext}"
                if audio_file.exists():
                    track_data['file_path'] = str(audio_file)
                    logger.info(f"Found audio file: {audio_file}")
                    break
            
            if 'file_path' not in track_data:
                logger.error(f"Could not find audio file for {track_name}")
                continue
        
        # Get segment types
        track_segments = track_data['segments']
        intros = [s for s in track_segments if s.get('type') == 'intro']
        verses = [s for s in track_segments if s.get('type') == 'verse']
        choruses = [s for s in track_segments if s.get('type') == 'chorus']
        bridges = [s for s in track_segments if s.get('type') == 'bridge']
        outros = [s for s in track_segments if s.get('type') == 'outro']
        
        # Pick a segment based on position in sequence
        segment = None
        position = i / len(track_sequence)
        
        # First appearance of a track
        if track_name not in [s.get('track') for s in segments]:
            if intros:
                segment = random.choice(intros)
            elif verses:
                segment = random.choice(verses)
        # Middle positions - mix of verses, choruses, bridges
        elif position < 0.7:
            if i % 2 == 0 and verses:
                segment = random.choice(verses)
            elif choruses:
                segment = random.choice(choruses)
            elif bridges:
                segment = random.choice(bridges)
        # Final appearance of a track
        elif i == len(track_sequence) - 1:
            if outros:
                segment = random.choice(outros)
            elif choruses:
                segment = random.choice(choruses)
        # Other positions
        else:
            segment_types = []
            if verses: segment_types.append(verses)
            if choruses: segment_types.append(choruses)
            if bridges: segment_types.append(bridges)
            
            if segment_types:
                segment_list = random.choice(segment_types)
                segment = random.choice(segment_list)
        
        # If still no segment, just pick any
        if not segment and track_segments:
            segment = random.choice(track_segments)
        elif not segment:
            logger.error(f"No usable segments for {track_name}")
            continue
        
        # Make sure the segment has the necessary fields
        if 'id' not in segment:
            current_segment_ids[track_name] += 1
            segment['id'] = f"{track_name}_{current_segment_ids[track_name]}"
        if 'track' not in segment:
            segment['track'] = track_name
        
        # Add segment to the mashup
        segments.append(segment)
        
        # Create a transition if this isn't the first segment
        if len(segments) > 1:
            # Create more interesting transitions based on segment types
            from_segment = segments[-2]
            to_segment = segments[-1]
            
            # Determine transition type based on segment characteristics
            if from_segment['track'] == to_segment['track']:
                # Same track transitions are simpler
                transition_type = random.choice(['cut', 'crossfade'])
                duration = random.uniform(0.5, 1.5)
            else:
                # Different track transitions are more complex
                if from_segment.get('type') == 'chorus' and to_segment.get('type') == 'intro':
                    transition_type = random.choice(['beatmatch_crossfade', 'filter_sweep', 'echo_out'])
                    duration = random.uniform(2.5, 4.0)
                elif from_segment.get('type') == 'outro' and to_segment.get('type') in ['intro', 'verse']:
                    transition_type = random.choice(['spinup', 'reverse_cymbal', 'tape_stop'])
                    duration = random.uniform(2.0, 3.5)
                else:
                    transition_type = random.choice(['crossfade', 'beatmatch_crossfade', 'filter_sweep', 'scratch_in'])
                    duration = random.uniform(1.8, 3.2)
            
            transition = {
                'from_segment': from_segment['id'],
                'to_segment': to_segment['id'],
                'type': transition_type,
                'duration': duration,
                'score': 0.8
            }
            transitions.append(transition)
    
    # If we have at least 2 segments, we can create a mashup
    if len(segments) < 2:
        logger.error("Could not create mashup with at least 2 segments")
        return
    
    # Print mashup structure
    print("\nMashup Structure:")
    total_duration = 0
    previous_track = None
    
    for i, segment in enumerate(segments):
        current_track = segment['track']
        is_new_track = previous_track != current_track
        
        # Print track name for first segment or when switching tracks
        if is_new_track:
            print(f"\n  Track: {current_track}")
        
        print(f"    Segment {i+1}: {segment['type']} - {segment['duration']:.2f}s")
        total_duration += segment['duration']
        previous_track = current_track
        
        # Add transition if not the last segment
        if i < len(transitions):
            transition = transitions[i]
            print(f"      ↓ [{transition['type']}] ({transition['duration']:.2f}s)")
            total_duration += transition['duration']
    
    print(f"\nTotal mashup duration: {total_duration:.2f}s ({total_duration/60:.2f} minutes)")
    
    # Render the mashup
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"simple_mashup_{timestamp}.wav"
    
    logger.info(f"Rendering mashup to {output_path}")
    try:
        final_output_path = render_mashup(tracks_data, segments, transitions, output_path=output_path)
        logger.info(f"Mashup rendered to {final_output_path}")
        print(f"\nMashup saved to: {final_output_path}")
    except Exception as e:
        logger.error(f"Error rendering mashup: {str(e)}")
        print(f"\nError rendering mashup: {str(e)}")

def create_fallback_segments(track_data, track_name, duration):
    """Create basic segments for a track"""
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
            'bpm': track_data.get('bpm', 120),
            'key': track_data.get('key', 'C major')
        })
    
    track_data['segments'] = segments
    return segments

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        logger.error(f"Unexpected error: {str(e)}", exc_info=True) 