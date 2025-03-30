import os
import sys
import time
from pathlib import Path
import argparse
import json
import random
import traceback

from transitions_ai.src.audio_analysis import analyze_directory
from transitions_ai.src.phrase_segmentation import segment_track
from transitions_ai.src.transition_engine import build_transition_graph
from transitions_ai.src.mashup_generator import generate_mashup
from transitions_ai.src.audio_renderer import render_mashup
from transitions_ai.src.logger import get_logger
from transitions_ai.src.config import (
    INPUT_DIR, OUTPUT_DIR, TEMP_DIR, MAX_MASHUP_LENGTH_MINUTES,
    MIN_SONGS_IN_MASHUP, MAX_CONSECUTIVE_PHRASES_FROM_SAME_SONG
)

logger = get_logger("main")

def main():
    """Main entry point for the Transitions AI system"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Transitions AI - DJ Mashup Generator')
    parser.add_argument('--input', type=str, default=str(INPUT_DIR),
                        help='Input directory containing audio files')
    parser.add_argument('--output', type=str, default=str(OUTPUT_DIR),
                        help='Output directory for mashups')
    parser.add_argument('--duration', type=float, default=MAX_MASHUP_LENGTH_MINUTES,
                        help='Target mashup duration in minutes')
    parser.add_argument('--analyze-only', action='store_true',
                        help='Only perform audio analysis and segmentation')
    parser.add_argument('--random-vibes', action='store_true',
                        help='Focus on creativity and randomness over technical compatibility')
    parser.add_argument('--load-analysis', action='store_true',
                        help='Load existing analysis data instead of reanalyzing')
    parser.add_argument('--min-tracks', type=int, default=MIN_SONGS_IN_MASHUP,
                        help='Minimum number of tracks to include in mashup')
    parser.add_argument('--max-consecutive', type=int, default=MAX_CONSECUTIVE_PHRASES_FROM_SAME_SONG,
                        help='Maximum consecutive segments from the same track')
    parser.add_argument('--start-track', type=str, help='Track to start the mashup with')
    parser.add_argument('--end-track', type=str, help='Track to end the mashup with')
    parser.add_argument('--output-name', type=str, help='Name for output mashup file (without extension)')
    parser.add_argument('--verbose', '-v', action='count', default=0, help='Increase verbosity')
    parser.add_argument('--force', '-f', action='store_true', help='Force overwrite of existing output files')
    args = parser.parse_args()
    
    logger.info("Starting Transitions AI")
    logger.info(f"Input directory: {args.input}")
    logger.info(f"Output directory: {args.output}")
    
    # Adjust randomness based on flags
    randomness_boost = 1.0
    if args.random_vibes:
        randomness_boost = 1.5
        logger.info("Random vibes mode activated - prioritizing creativity over technical correctness!")
    
    # Create directories if they don't exist
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    temp_dir = Path(TEMP_DIR)
    
    for directory in [output_dir, temp_dir]:
        try:
            directory.mkdir(exist_ok=True, parents=True)
        except Exception as e:
            logger.error(f"Failed to create directory {directory}: {str(e)}")
            return
    
    # Validate input directory
    if not input_dir.exists():
        logger.error(f"Input directory {input_dir} does not exist.")
        return
    
    # Check for audio files
    audio_files = []
    for ext in ['.wav', '.mp3', '.flac', '.ogg', '.m4a']:
        audio_files.extend(list(input_dir.glob(f"*{ext}")))
    
    if not audio_files:
        logger.error(f"No audio files found in {input_dir}")
        return
    
    # Setup output filename
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_name = args.output_name if args.output_name else f"mashup_{timestamp}"
    output_path = output_dir / f"{output_name}.wav"
    
    # Check if output file already exists
    if output_path.exists() and not args.force:
        logger.error(f"Output file {output_path} already exists. Use --force to overwrite.")
        return
    
    # Step 1: Analyze audio files or load existing analysis
    tracks_data = {}
    data_dir = output_dir / "data"
    data_dir.mkdir(exist_ok=True, parents=True)
    
    if args.load_analysis and data_dir.exists():
        logger.info("Loading existing analysis data")
        try:
            for data_file in data_dir.glob("*.json"):
                try:
                    with open(data_file, 'r') as f:
                        track_data = json.load(f)
                        track_name = data_file.stem
                        tracks_data[track_name] = track_data
                        logger.info(f"Loaded analysis data for {track_name}")
                except Exception as e:
                    logger.warning(f"Error loading {data_file}: {str(e)}")
            
            if not tracks_data:
                logger.warning("No valid analysis data found, falling back to fresh analysis")
        except Exception as e:
            logger.error(f"Error loading analysis data: {str(e)}")
            logger.info("Falling back to fresh analysis")
    
    # Perform fresh analysis if needed
    if not tracks_data:
        logger.info(f"Analyzing audio files in {args.input}")
        try:
            track_features = analyze_directory(args.input)
            
            if not track_features:
                logger.error("No audio files found or analysis failed")
                return
            
            logger.info(f"Successfully analyzed {len(track_features)} tracks")
            
            # Step 2: Segment tracks
            logger.info("Segmenting tracks into musical phrases")
            
            for track_name, features in track_features.items():
                try:
                    segments = segment_track(features, track_name)
                    if not segments:
                        logger.warning(f"No segments returned for {track_name}, using fallback segmentation")
                        segments = _create_fallback_segments(track_name, features)
                    
                    tracks_data[track_name] = {
                        **features,  # Include all audio features
                        'segments': segments  # Add segments
                    }
                    
                    # Print segment information
                    if args.verbose > 0:
                        _print_track_info(track_name, features, segments)
                    
                except Exception as e:
                    logger.error(f"Error segmenting {track_name}: {str(e)}")
                    # Create basic fallback segments
                    try:
                        segments = _create_fallback_segments(track_name, features)
                        
                        tracks_data[track_name] = {
                            **features,
                            'segments': segments
                        }
                        
                        logger.info(f"Created {len(segments)} emergency fallback segments for {track_name}")
                        
                        if args.verbose > 0:
                            _print_track_info(track_name, features, segments, fallback=True)
                    except Exception as inner_e:
                        logger.error(f"Failed to create fallback segments for {track_name}: {str(inner_e)}")
            
            # Save track data for future use
            _save_track_data(tracks_data, data_dir, args.input)
            
        except Exception as e:
            logger.error(f"Fatal error during analysis phase: {str(e)}")
            return
    
    # Exit if only performing analysis
    if args.analyze_only:
        logger.info("Analysis complete. Exiting.")
        return
    
    # Step 3: Build transition graph
    logger.info("Building transition compatibility graph")
    try:
        transition_matrix = build_transition_graph(tracks_data, randomness_boost=randomness_boost)
        logger.info(f"Built transition matrix with {sum(len(v) for v in transition_matrix.values())} possible transitions")
    except Exception as e:
        logger.error(f"Error building transition graph: {str(e)}")
        transition_matrix = {}  # Empty transition matrix
    
    # Step 4: Generate mashup
    logger.info(f"Generating mashup (target duration: {args.duration} minutes)")
    try:
        # First print information about available tracks and segments
        logger.info("Available tracks for mashup generation:")
        for track_name, track_data in tracks_data.items():
            if 'segments' in track_data and track_data['segments']:
                logger.info(f"  Track: {track_name} - {len(track_data['segments'])} segments")
                if args.verbose > 1:  # Extra verbose output
                    for i, segment in enumerate(track_data['segments']):
                        logger.info(f"    Segment {i+1}: {segment['type']} {segment['start']:.1f}s-{segment['end']:.1f}s")
            else:
                logger.warning(f"  Track: {track_name} - NO SEGMENTS FOUND")
        
        # Print information about transitions
        logger.info("Transition matrix statistics:")
        if transition_matrix:
            segment_count = len(transition_matrix)
            transition_count = sum(len(v) for v in transition_matrix.values())
            logger.info(f"  {segment_count} segments with {transition_count} possible transitions")
            
            # Check if any segments have no outgoing transitions
            segments_with_no_transitions = [seg_id for seg_id, transitions in transition_matrix.items() if not transitions]
            if segments_with_no_transitions:
                logger.warning(f"  {len(segments_with_no_transitions)} segments have no outgoing transitions")
                
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
                logger.info(f"  Track {track} can transition to: {', '.join(connected_tracks) if connected_tracks else 'NONE'}")
        else:
            logger.warning("  No transitions found in transition matrix!")

        # Generate the mashup
        mashup_segments, mashup_transitions = generate_mashup(
            tracks_data, 
            transition_matrix,
            target_duration=args.duration,
            start_track=args.start_track,
            end_track=args.end_track,
            min_tracks=args.min_tracks,
            max_consecutive=args.max_consecutive,
            randomness_boost=randomness_boost
        )
        
        if not mashup_segments:
            logger.error("Failed to generate mashup - no segments returned")
            return
        
        logger.info(f"Generated mashup with {len(mashup_segments)} segments and {len(mashup_transitions)} transitions")
        
        # Check which tracks are included in the mashup
        tracks_in_mashup = set(segment['track'] for segment in mashup_segments)
        logger.info(f"Tracks in mashup: {', '.join(tracks_in_mashup)}")
        
        # Print mashup structure
        if args.verbose > 0:
            _print_mashup_structure(mashup_segments, mashup_transitions)
            
    except Exception as e:
        logger.error(f"Error generating mashup: {str(e)}")
        logger.error(traceback.format_exc())
        return
    
    # Step 5: Render audio mashup
    logger.info("Rendering audio mashup")
    try:
        final_output_path = render_mashup(tracks_data, mashup_segments, mashup_transitions, output_path=output_path)
        logger.info(f"Mashup rendered to {final_output_path}")
        print(f"\nMashup saved to: {final_output_path}")
        
        # Also save mashup structure
        try:
            mashup_data = {
                "segments": mashup_segments,
                "transitions": mashup_transitions,
                "created_at": timestamp,
                "duration": sum(seg["duration"] for seg in mashup_segments) + 
                           sum(trans["duration"] for trans in mashup_transitions)
            }
            
            with open(output_dir / f"{output_name}.json", 'w') as f:
                json.dump(mashup_data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save mashup structure data: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error rendering mashup: {str(e)}")
        print(f"\nError rendering mashup: {str(e)}")
    
    logger.info("Transitions AI process complete")

def _create_fallback_segments(track_name, features):
    """Create fallback segments for a track when segmentation fails"""
    duration = features['duration']
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
            'bpm': features.get('bpm', 120),
            'key': features.get('key', 'C major')
        })
    
    return segments

def _print_track_info(track_name, features, segments, fallback=False):
    """Print information about a track and its segments"""
    print(f"\nTrack: {track_name}")
    print(f"  BPM: {features['bpm']:.1f}")
    print(f"  Key: {features['key']}")
    print(f"  Duration: {features['duration']:.2f}s")
    print(f"  Segments: {len(segments)}{' (fallback)' if fallback else ''}")
    
    # Print segment details
    for i, segment in enumerate(segments):
        print(f"    {i+1}. {segment['type']}: "
              f"{segment['start']:.2f}s - {segment['end']:.2f}s "
              f"(duration: {segment['duration']:.2f}s)")

def _save_track_data(tracks_data, data_dir, input_dir_path):
    """Save track data for future use"""
    try:
        input_path = Path(input_dir_path)
        
        for track_name, data in tracks_data.items():
            # Add file path for future reference
            for ext in ['.wav', '.mp3', '.flac', '.ogg', '.m4a']:
                audio_file = input_path / f"{track_name}{ext}"
                if audio_file.exists():
                    data['file_path'] = str(audio_file)
                    break
            
            # Convert numpy arrays to lists for JSON serialization
            serializable_data = {}
            for key, value in data.items():
                if key != 'segments':
                    if hasattr(value, 'tolist'):
                        serializable_data[key] = value.tolist()
                    else:
                        serializable_data[key] = value
                else:
                    serializable_data[key] = value
            
            # Save track data
            with open(data_dir / f"{track_name}.json", 'w') as f:
                json.dump(serializable_data, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving track data: {str(e)}")

def _print_mashup_structure(segments, transitions):
    """Print the structure of a generated mashup"""
    print("\nMashup Structure:")
    total_duration = 0
    for i, segment in enumerate(segments):
        print(f"  {i+1}. {segment['track']} - {segment['type']}: {segment['duration']:.2f}s")
        total_duration += segment['duration']
        
        # Add transition if not the last segment
        if i < len(transitions):
            transition = transitions[i]
            print(f"     ↓ [{transition['type']}] ({transition['duration']:.2f}s, score: {transition['score']:.2f})")
            total_duration += transition['duration']
    
    print(f"\nTotal mashup duration: {total_duration:.2f}s ({total_duration/60:.2f} minutes)")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        logger.info("Process interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        sys.exit(1) 