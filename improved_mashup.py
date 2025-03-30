import os
import sys
import time
import argparse
from pathlib import Path
import json
import random
import math

from transitions_ai.src.config import (
    INPUT_DIR, OUTPUT_DIR, TEMP_DIR
)
from transitions_ai.src.logger import get_logger
from transitions_ai.src.audio_renderer import render_mashup

logger = get_logger("improved_mashup")

def main():
    """
    Creates a DJ-style mashup that respects musical structure and creates smooth transitions
    at natural boundaries, with a focus on musical coherence and flow.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Create a DJ-style mashup with smooth transitions')
    parser.add_argument('--output-name', type=str, help='Name for the output file')
    parser.add_argument('--track-sequence', type=str, help='Comma-separated list of tracks in desired order')
    parser.add_argument('--transition-length', type=float, default=8.0, 
                        help='Default transition length in seconds (default: 8.0)')
    parser.add_argument('--min-transition', type=float, default=4.0,
                        help='Minimum transition length in seconds (default: 4.0)')
    parser.add_argument('--max-transition', type=float, default=16.0,
                        help='Maximum transition length in seconds (default: 16.0)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show verbose output')
    parser.add_argument('--transition-types', type=str, default='beatmatch_crossfade,filter_sweep,echo_out',
                        help='Comma-separated list of preferred transition types')
    parser.add_argument('--segment-types', type=str, default='chorus,verse,intro,bridge,outro',
                        help='Comma-separated list of segment types to include in order of preference')
    parser.add_argument('--full-segments', action='store_true', default=True,
                        help='Always use complete segments without trimming')
    parser.add_argument('--energy-flow', type=str, choices=['increase', 'decrease', 'wave', 'random'], 
                        default='wave', help='Energy flow pattern for the mashup')
    args = parser.parse_args()
    
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
                
                # Ensure track has segments
                if 'segments' not in track_data or not track_data['segments']:
                    logger.warning(f"No segments found for {track_name}, creating basic segments")
                    duration = track_data.get('duration', 240.0)
                    create_fallback_segments(track_data, track_name, duration)
                
                # Ensure segments have track field
                for segment in track_data.get('segments', []):
                    if 'track' not in segment:
                        segment['track'] = track_name
                
                # Enhance segment information with better timing data
                enhance_segments(track_data)
                
                # Ensure file path is set
                if 'file_path' not in track_data:
                    for ext in ['.wav', '.mp3', '.flac', '.ogg', '.m4a']:
                        audio_file = Path(INPUT_DIR) / f"{track_name}{ext}"
                        if audio_file.exists():
                            track_data['file_path'] = str(audio_file)
                            break
                
                tracks_data[track_name] = track_data
                logger.info(f"Loaded track {track_name} with {len(track_data.get('segments', []))} segments")
        except Exception as e:
            logger.error(f"Error loading {data_file}: {str(e)}")
    
    # Make sure we have at least 2 tracks
    if len(tracks_data) < 2:
        logger.error(f"Need at least 2 tracks for a mashup, but only found {len(tracks_data)}.")
        return
    
    # Get track names and print available tracks
    track_names = list(tracks_data.keys())
    logger.info(f"Available tracks: {', '.join(track_names)}")
    print("Available tracks:")
    for i, track_name in enumerate(track_names):
        track_data = tracks_data[track_name]
        print(f"  {i+1}. {track_name}")
        print(f"     BPM: {track_data.get('bpm', 'unknown')}")
        print(f"     Key: {track_data.get('key', 'unknown')}")
        print(f"     Duration: {track_data.get('duration', 0):.2f}s")
        
        # Print segment info if verbose
        if args.verbose:
            segments = track_data.get('segments', [])
            print(f"     Segments: {len(segments)}")
            for j, segment in enumerate(segments):
                print(f"       {j+1}. {segment.get('type', 'unknown')}: {segment.get('start', 0):.1f}s-{segment.get('end', 0):.1f}s ({segment.get('duration', 0):.1f}s)")
        else:
            print(f"     Segments: {len(track_data.get('segments', []))}")
    
    # Parse preferred transition types
    preferred_transitions = [t.strip() for t in args.transition_types.split(',') if t.strip()]
    logger.info(f"Preferred transition types: {', '.join(preferred_transitions)}")
    
    # Parse preferred segment types
    preferred_segments = [s.strip() for s in args.segment_types.split(',') if s.strip()]
    logger.info(f"Preferred segment types: {', '.join(preferred_segments)}")
    
    # Determine track sequence
    selected_tracks = []
    if args.track_sequence:
        # Parse user-specified track sequence
        try:
            track_indices = [int(i.strip()) - 1 for i in args.track_sequence.split(',')]
            selected_tracks = [track_names[i] for i in track_indices if 0 <= i < len(track_names)]
        except (ValueError, IndexError):
            # Try matching by name
            selected_tracks = [t for t in args.track_sequence.split(',') if t.strip() in track_names]
            
        if not selected_tracks:
            logger.warning("Invalid track sequence. Using all tracks in a balanced order.")
            selected_tracks = track_names.copy()
    else:
        # Use all tracks with smart ordering based on musical key and BPM
        selected_tracks = order_tracks_by_compatibility(tracks_data)
    
    logger.info(f"Selected tracks for mashup: {', '.join(selected_tracks)}")
    
    # Create DJ-oriented track sequence that creates a smooth flow
    track_sequence = create_dj_sequence(selected_tracks)
    
    # Build the segments and transitions
    params = {
        'transition_length': args.transition_length,
        'min_transition': args.min_transition,
        'max_transition': args.max_transition,
        'preferred_transitions': preferred_transitions,
        'preferred_segments': preferred_segments,
        'full_segments': args.full_segments,
        'energy_flow': args.energy_flow,
    }
    
    segments, transitions = build_mashup(tracks_data, track_sequence, params)
    
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
        
        print(f"    Segment {i+1}: {segment['type']} - {segment['duration']:.2f}s (Energy: {segment.get('avg_energy', 0.5):.2f})")
        total_duration += segment['duration']
        previous_track = current_track
        
        # Add transition if not the last segment
        if i < len(transitions):
            transition = transitions[i]
            energy_change = transition.get('to_energy', 0) - transition.get('from_energy', 0)
            energy_change_str = f"(energy change: {energy_change:.2f})"
            explanation = transition.get('explanation', '')
            print(f"      ↓ [{transition['type']}] ({transition['duration']:.2f}s) {energy_change_str}")
            print(f"         {explanation}")
            total_duration += transition['duration']
    
    print(f"\nTotal mashup duration: {total_duration:.2f}s ({total_duration/60:.2f} minutes)")
    
    # Render the mashup
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_name = args.output_name if args.output_name else f"dj_mashup_{timestamp}"
    output_path = output_dir / f"{output_name}.wav"
    
    logger.info(f"Rendering mashup to {output_path}")
    try:
        final_output_path = render_mashup(tracks_data, segments, transitions, output_path=output_path)
        logger.info(f"Mashup rendered to {final_output_path}")
        print(f"\nMashup saved to: {final_output_path}")
    except Exception as e:
        logger.error(f"Error rendering mashup: {str(e)}")
        print(f"\nError rendering mashup: {str(e)}")

def order_tracks_by_compatibility(tracks_data):
    """Order tracks by musical compatibility (key and BPM)"""
    track_names = list(tracks_data.keys())
    
    # If only 2 tracks, just return them
    if len(track_names) <= 2:
        return track_names
    
    # Get track features
    track_features = {}
    for name in track_names:
        track = tracks_data[name]
        track_features[name] = {
            'bpm': float(track.get('bpm', 120)),
            'key': track.get('key', 'C major')
        }
    
    # Start with a track with moderate BPM
    bpms = [(name, feature['bpm']) for name, feature in track_features.items()]
    bpms.sort(key=lambda x: x[1])
    
    # Start with a track in the middle of the BPM range
    middle_idx = len(bpms) // 2
    ordered = [bpms[middle_idx][0]]
    remaining = [name for name in track_names if name != ordered[0]]
    
    # Build the sequence to have smooth BPM progressions
    while remaining:
        last_track = ordered[-1]
        last_bpm = track_features[last_track]['bpm']
        
        # Find the closest BPM track
        closest = min(remaining, key=lambda x: abs(track_features[x]['bpm'] - last_bpm))
        ordered.append(closest)
        remaining.remove(closest)
    
    return ordered

def create_dj_sequence(track_names):
    """Create a DJ-friendly sequence that repeats tracks for a better mix"""
    if len(track_names) <= 1:
        return track_names
    
    # For 2 tracks, alternate them a few times
    if len(track_names) == 2:
        return [track_names[0], track_names[1], track_names[0], track_names[1], track_names[0]]
    
    # For 3+ tracks, create a sequence with some repetition for cohesion
    sequence = []
    
    # Start with first track
    sequence.append(track_names[0])
    
    # Add all tracks in order
    sequence.extend(track_names[1:])
    
    # Return to first track
    sequence.append(track_names[0])
    
    # Add second half of tracks in reverse order for a good arc
    if len(track_names) >= 4:
        second_half = track_names[len(track_names)//2:]
        sequence.extend(reversed(second_half))
    else:
        # For 3 tracks, add a simple ending
        sequence.append(track_names[-1])
    
    return sequence

def enhance_segments(track_data):
    """Add additional properties to segments for better transitions"""
    segments = track_data.get('segments', [])
    
    # Sort segments by start time
    segments.sort(key=lambda s: s.get('start', 0))
    
    # Enhance segments with derived properties
    for i, segment in enumerate(segments):
        # Ensure segment has all required fields
        if 'start' not in segment or 'end' not in segment:
            continue
            
        # Calculate downbeats (assuming 4/4 time)
        bpm = track_data.get('bpm', 120)
        beats_per_second = bpm / 60
        beats_in_segment = (segment['end'] - segment['start']) * beats_per_second
        measures = int(beats_in_segment / 4)
        
        # Add musical context information
        segment['measures'] = measures
        segment['beat_count'] = int(beats_in_segment)
        segment['is_downbeat_start'] = True  # Assume all segments start on downbeats
        segment['bpm'] = bpm  # Add BPM to segment
        
        # Calculate rough energy level if not already present
        if 'avg_energy' not in segment:
            # Assign energy based on segment type
            if segment['type'] == 'chorus':
                segment['avg_energy'] = 0.8
            elif segment['type'] == 'verse':
                segment['avg_energy'] = 0.6
            elif segment['type'] == 'bridge':
                segment['avg_energy'] = 0.5
            elif segment['type'] == 'intro':
                segment['avg_energy'] = 0.4
            elif segment['type'] == 'outro':
                segment['avg_energy'] = 0.3
            else:
                segment['avg_energy'] = 0.5
    
    return segments

def build_mashup(tracks_data, track_sequence, params):
    """Build a mashup with DJ-friendly transitions that respect musical structure"""
    segments = []
    transitions = []
    
    # Extract parameters
    transition_length = params.get('transition_length', 8.0)
    min_transition = params.get('min_transition', 4.0)
    max_transition = params.get('max_transition', 16.0)
    preferred_transitions = params.get('preferred_transitions', ['beatmatch_crossfade'])
    preferred_segments = params.get('preferred_segments', ['chorus', 'verse', 'intro', 'bridge', 'outro'])
    full_segments = params.get('full_segments', True)
    energy_flow = params.get('energy_flow', 'wave')
    
    # Track the overall energy curve
    target_energy_levels = []
    
    # Set up energy flow based on chosen pattern
    if energy_flow == 'increase':
        # Gradually increasing energy
        for i in range(len(track_sequence)):
            position = i / len(track_sequence)
            target_energy_levels.append(0.3 + position * 0.6)  # 0.3 to 0.9
    elif energy_flow == 'decrease':
        # Gradually decreasing energy
        for i in range(len(track_sequence)):
            position = i / len(track_sequence)
            target_energy_levels.append(0.9 - position * 0.6)  # 0.9 to 0.3
    elif energy_flow == 'wave':
        # Up and down wave pattern
        for i in range(len(track_sequence)):
            position = i / len(track_sequence)
            # Sine wave oscillating between 0.4 and 0.8
            energy = 0.6 + 0.2 * math.sin(position * 2 * math.pi)
            target_energy_levels.append(energy)
    else:  # random
        # Random energy levels
        for i in range(len(track_sequence)):
            target_energy_levels.append(random.uniform(0.4, 0.8))
            
    logger.info(f"Energy flow pattern: {energy_flow}")
    
    # Track used segments to avoid repeating the same ones
    used_segments = set()
    
    # First pass: analyze the whole track sequence to plan segment selection
    planned_segments = []
    
    # For each track in the sequence, select segments intelligently
    for i, track_name in enumerate(track_sequence):
        track_data = tracks_data.get(track_name)
        if not track_data or 'segments' not in track_data:
            logger.warning(f"Missing data for track {track_name}")
            continue
        
        track_segments = track_data['segments']
        if not track_segments:
            logger.warning(f"No segments for track {track_name}")
            continue
        
        # Sort segments by position and label them with index
        track_segments = sorted(track_segments, key=lambda s: s.get('start', 0))
        for j, segment in enumerate(track_segments):
            segment['idx'] = j
            segment['total'] = len(track_segments)
        
        # Check if this is a new track or we've seen it before
        first_appearance = track_name not in [s.get('track') for s in planned_segments]
        last_appearance = i == len(track_sequence) - 1
        
        # Get the target energy for this track
        target_energy = target_energy_levels[i % len(target_energy_levels)]
        
        # Group segments by type
        segment_by_type = {}
        for segment_type in set(s.get('type', 'unknown') for s in track_segments):
            segment_by_type[segment_type] = [s for s in track_segments if s.get('type') == segment_type]
        
        # Select segments based on position in mix
        selected_segments = []
        
        if first_appearance:
            # First appearance: preferably use intro or first verse, never start with chorus
            intros = segment_by_type.get('intro', [])
            verses = segment_by_type.get('verse', [])
            
            if intros and intros[0].get('avg_energy', 0.5) < 0.7:  # Only use intro if not too energetic
                selected_segments.append(intros[0])
                
                # Add first verse too if intro is short
                if intros[0].get('duration', 0) < 20 and verses:
                    # Add first verse
                    selected_segments.append(verses[0])
            elif verses:
                # Use first verse
                selected_segments.append(verses[0])
            elif track_segments:
                # Use first segment but not chorus
                for segment in track_segments:
                    if segment.get('type') != 'chorus':
                        selected_segments.append(segment)
                        break
                
                # If no suitable segment found, use first segment
                if not selected_segments:
                    selected_segments.append(track_segments[0])
        
        elif last_appearance:
            # Last appearance: use chorus followed by outro, or ending segments
            outros = segment_by_type.get('outro', [])
            choruses = segment_by_type.get('chorus', [])
            
            # For ending: chorus -> outro sequence is ideal
            if choruses and outros:
                selected_segments.append(choruses[-1])  # Last chorus
                selected_segments.append(outros[0])     # Outro
            elif outros:
                selected_segments.append(outros[0])
            elif choruses:
                selected_segments.append(choruses[-1])
            elif track_segments:
                # Use last quarter of track
                quarter_idx = len(track_segments) * 3 // 4
                selected_segments.extend(track_segments[quarter_idx:])
        
        else:
            # Middle appearance: select segments to maintain flow and avoid energy killers
            
            # Find good chorus/verse boundaries
            good_segments = []
            
            # Prefer chorus -> verse or verse -> chorus transitions (good DJ transition points)
            for j in range(len(track_segments) - 1):
                seg1 = track_segments[j]
                seg2 = track_segments[j+1]
                
                # Good transition points
                good_transition = (
                    (seg1.get('type') == 'chorus' and seg2.get('type') == 'verse') or
                    (seg1.get('type') == 'verse' and seg2.get('type') == 'chorus')
                )
                
                if good_transition:
                    # Calculate how close the segment is to target energy
                    energy_match = abs(seg1.get('avg_energy', 0.5) - target_energy)
                    good_segments.append((seg1, energy_match))
            
            # If we found good transition segments, use the one closest to target energy
            if good_segments:
                good_segments.sort(key=lambda x: x[1])  # Sort by energy match
                selected_segments.append(good_segments[0][0])
            else:
                # Find segments close to target energy, avoiding high-energy segments that aren't chorus
                best_segments = []
                for segment in track_segments:
                    # Skip segments that would kill energy
                    if segment.get('type') == 'outro':
                        continue
                    
                    energy_match = abs(segment.get('avg_energy', 0.5) - target_energy)
                    
                    # Penalize high-energy segments that aren't chorus
                    if segment.get('avg_energy', 0.5) > 0.7 and segment.get('type') != 'chorus':
                        energy_match += 0.3
                    
                    # Penalize cutting in the middle of segments
                    position = segment.get('idx', 0) / max(1, segment.get('total', 1))
                    if 0.3 < position < 0.7:  # Middle segments
                        energy_match += 0.2
                    
                    best_segments.append((segment, energy_match))
                
                if best_segments:
                    best_segments.sort(key=lambda x: x[1])  # Sort by energy match
                    selected_segments.append(best_segments[0][0])
                    
                    # Maybe add another segment that follows naturally
                    if len(track_segments) > 1:
                        idx = best_segments[0][0].get('idx', 0)
                        next_idx = idx + 1
                        
                        if next_idx < len(track_segments):
                            selected_segments.append(track_segments[next_idx])
        
        # If no segments were selected, use segments with good energy
        if not selected_segments and track_segments:
            # Select segment with energy closest to target
            best_match = min(track_segments, 
                           key=lambda s: abs(s.get('avg_energy', 0.5) - target_energy))
            selected_segments.append(best_match)
        
        # Sort selected segments by their position in the track
        selected_segments.sort(key=lambda s: s.get('start', 0))
        
        # Add segments to planned segments
        for segment in selected_segments:
            if segment.get('id') not in used_segments:
                planned_segments.append(segment)
                used_segments.add(segment.get('id'))
    
    # Second pass: create transitions between planned segments
    segments = planned_segments
    
    # Create transitions between segments
    for i in range(len(segments) - 1):
        from_segment = segments[i]
        to_segment = segments[i+1]
        
        # Only create transitions between segments from different tracks
        if from_segment['track'] != to_segment['track']:
            # Calculate a transition that makes musical sense
            transition = create_musical_transition(
                from_segment, to_segment, 
                transition_length=transition_length,
                min_transition=min_transition,
                max_transition=max_transition,
                preferred_transitions=preferred_transitions
            )
            transitions.append(transition)
    
    return segments, transitions

def create_musical_transition(from_segment, to_segment, transition_length=8.0, 
                              min_transition=4.0, max_transition=16.0, 
                              preferred_transitions=None):
    """Create a transition that respects musical structure and energy flow"""
    # Determine the best transition type based on segment types and energy
    from_type = from_segment.get('type', 'unknown')
    to_type = to_segment.get('type', 'unknown')
    from_energy = from_segment.get('avg_energy', 0.5)
    to_energy = to_segment.get('avg_energy', 0.5)
    
    # Default preferred transitions
    if not preferred_transitions:
        preferred_transitions = ['beatmatch_crossfade', 'filter_sweep', 'echo_out']
    
    # Calculate a musically appropriate transition length (multiple of 4 beats typically)
    bpm = from_segment.get('bpm', 120)
    beats_per_second = bpm / 60
    
    # Base transition on musical time (typically 8, 16, or 32 beats)
    beat_counts = [8, 16, 32]  # Common transition lengths in beats
    
    # Choose beat count based on energy difference
    energy_diff = abs(to_energy - from_energy)
    if energy_diff > 0.3:
        # Bigger energy change = longer transition
        beat_count = 32
    elif energy_diff > 0.15:
        beat_count = 16
    else:
        beat_count = 8
    
    # Make sure transition isn't longer than half the shorter segment
    min_segment_duration = min(from_segment.get('duration', 30), to_segment.get('duration', 30))
    max_transition_duration = min(max_transition, min_segment_duration * 0.4)  # Max 40% of segment duration
    
    # Calculate duration based on beats, but constrain to limits
    duration = beat_count / beats_per_second
    duration = max(min_transition, min(max_transition_duration, duration))
    
    # Choose transition type based on energy flow and preferred transitions
    energy_diff = to_energy - from_energy
    transition_type = preferred_transitions[0]  # Default to first preferred
    
    # Add transition explanation
    transition_explanation = ""
    
    # Choose appropriate transition types
    candidates = []
    
    if energy_diff >= 0.2:  # Increasing energy
        # Prefer transitions that build energy
        energy_building = ['filter_sweep', 'spinup', 'drum_fill', 'reverse_cymbal']
        # Intersect with preferred transitions
        candidates = [t for t in energy_building if t in preferred_transitions]
        transition_explanation = "energy build"
    elif energy_diff <= -0.2:  # Decreasing energy
        # Prefer transitions that reduce energy
        energy_reducing = ['echo_out', 'tape_stop', 'filter_sweep']
        candidates = [t for t in energy_reducing if t in preferred_transitions]
        transition_explanation = "energy reduction"
    else:  # Similar energy
        # Prefer transitions that maintain energy
        energy_neutral = ['beatmatch_crossfade', 'harmonic_crossfade']
        candidates = [t for t in energy_neutral if t in preferred_transitions]
    
    # If we have candidates, use them, otherwise use any preferred transition
    if candidates:
        transition_type = random.choice(candidates)
    elif preferred_transitions:
        transition_type = random.choice(preferred_transitions)
    
    # Adjust transition description based on segment types
    segment_transition = f"{from_type}-to-{to_type}"
    
    # Create the transition object
    transition = {
        'from_segment': from_segment.get('id', ''),
        'to_segment': to_segment.get('id', ''),
        'type': transition_type,
        'duration': duration,
        'score': 0.9,  # High confidence score
        'explanation': f"{transition_explanation}, {segment_transition}",
        'from_energy': from_energy,
        'to_energy': to_energy
    }
    
    return transition

def create_fallback_segments(track_data, track_name, duration):
    """Create basic segments for a track"""
    num_segments = max(4, min(6, int(duration / 30)))
    segment_duration = duration / num_segments
    
    segments = []
    for i in range(num_segments):
        start = i * segment_duration
        end = min((i + 1) * segment_duration, duration)
        
        # Segment type based on position and energy curve
        if i == 0:
            segment_type = 'intro'
        elif i == num_segments - 1:
            segment_type = 'outro'
        elif i % 2 == 0:
            segment_type = 'verse'
        else:
            segment_type = 'chorus'
        
        # Calculate average energy based on segment type (for a typical energy arc)
        if segment_type == 'chorus':
            avg_energy = 0.8
        elif segment_type == 'verse':
            avg_energy = 0.6
        elif segment_type == 'intro':
            avg_energy = 0.4
        elif segment_type == 'outro':
            avg_energy = 0.3
        else:
            avg_energy = 0.5
        
        segments.append({
            'id': f"{track_name}_{i+1}",
            'track': track_name,
            'start': float(start),
            'end': float(end),
            'duration': float(end - start),
            'type': segment_type,
            'source': 'fallback',
            'avg_energy': avg_energy,
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