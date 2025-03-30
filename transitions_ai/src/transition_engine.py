import os
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
import traceback
import random

from transitions_ai.src.config import (
    TRANSITION_TYPES, DEFAULT_TRANSITION_TYPE, DEFAULT_TRANSITION_DURATION_SECONDS,
    MAX_BPM_DIFFERENCE, BPM_CHANGE_TOLERANCE, KEY_COMPATIBILITY_THRESHOLD,
    ENERGY_MATCHING_WEIGHT, HARMONIC_COMPATIBILITY_WEIGHT, RHYTHM_COMPATIBILITY_WEIGHT,
    RANDOMNESS_FACTOR, SURPRISE_TRANSITION_CHANCE, VIBE_PATTERN_WEIGHT,
    CREATIVE_WEIGHT
)
from transitions_ai.src.logger import get_logger, log_transition

logger = get_logger("transition")

# Define key compatibility chart (Camelot wheel)
KEY_COMPATIBILITY = {
    # Major keys
    'C major': ['C major', 'G major', 'A minor', 'E minor'],
    'G major': ['G major', 'D major', 'E minor', 'B minor'],
    'D major': ['D major', 'A major', 'B minor', 'F# minor'],
    'A major': ['A major', 'E major', 'F# minor', 'C# minor'],
    'E major': ['E major', 'B major', 'C# minor', 'G# minor'],
    'B major': ['B major', 'F# major', 'G# minor', 'D# minor'],
    'F# major': ['F# major', 'C# major', 'D# minor', 'A# minor'],
    'C# major': ['C# major', 'G# major', 'A# minor', 'F minor'],
    'G# major': ['G# major', 'D# major', 'F minor', 'C minor'],
    'D# major': ['D# major', 'A# major', 'C minor', 'G minor'],
    'A# major': ['A# major', 'F major', 'G minor', 'D minor'],
    'F major': ['F major', 'C major', 'D minor', 'A minor'],
    
    # Minor keys
    'A minor': ['A minor', 'E minor', 'C major', 'G major'],
    'E minor': ['E minor', 'B minor', 'G major', 'D major'],
    'B minor': ['B minor', 'F# minor', 'D major', 'A major'],
    'F# minor': ['F# minor', 'C# minor', 'A major', 'E major'],
    'C# minor': ['C# minor', 'G# minor', 'E major', 'B major'],
    'G# minor': ['G# minor', 'D# minor', 'B major', 'F# major'],
    'D# minor': ['D# minor', 'A# minor', 'F# major', 'C# major'],
    'A# minor': ['A# minor', 'F minor', 'C# major', 'G# major'],
    'F minor': ['F minor', 'C minor', 'G# major', 'D# major'],
    'C minor': ['C minor', 'G minor', 'D# major', 'A# major'],
    'G minor': ['G minor', 'D minor', 'A# major', 'F major'],
    'D minor': ['D minor', 'A minor', 'F major', 'C major'],
}

class TransitionEngine:
    """
    Engine for determining compatibility between track segments 
    and generating transition graphs
    """
    
    def __init__(self, randomness_boost=1.0):
        self.logger = get_logger("transition")
        self.logger.info("Initializing transition engine")
        self.transition_graph = {}
        self.vibe_patterns = {
            "energy_flow": [0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 0.7, 0.5],
            "energy_drop": [0.8, 0.7, 0.3, 0.2, 0.5, 0.7],
            "tension_release": [0.6, 0.7, 0.8, 0.9, 0.4, 0.3],
            "surprising_contrasts": [0.5, 0.5, 0.9, 0.2, 0.8, 0.3],
            "classic_flow": [0.4, 0.6, 0.7, 0.6, 0.8, 0.7],
            "random_vibes": [random.random() for _ in range(6)]  # Completely random pattern
        }
        # Add more randomness/variance based on the boost
        self.randomness_factor = RANDOMNESS_FACTOR * randomness_boost
        
        # Sometimes just completely randomize our approach for this mashup
        if random.random() < 0.3 * randomness_boost:
            self.randomness_factor *= 1.5
            self.logger.info(f"Choosing extra random approach! Randomness factor: {self.randomness_factor}")
            
    def build_transition_graph(self, segments_by_track: Dict[str, List[Dict[str, Any]]]) -> Dict:
        """
        Build a graph of possible transitions between segments from different tracks
        
        Args:
            segments_by_track: Dictionary mapping track names to segment lists
            
        Returns:
            Transition graph dictionary
        """
        self.logger.info("Building transition graph")
        
        # Flatten segments from all tracks
        all_segments = []
        for track_name, segments in segments_by_track.items():
            if not segments:
                self.logger.warning(f"No segments found for track {track_name}")
                continue
            all_segments.extend(segments)
        
        if not all_segments:
            self.logger.error("No segments found in any tracks")
            return {}
            
        self.logger.info(f"Building transitions for {len(all_segments)} segments from {len(segments_by_track)} tracks")
        
        # Choose a "vibe pattern" for this mashup
        current_vibe = random.choice(list(self.vibe_patterns.keys()))
        self.logger.info(f"Selected vibe pattern: {current_vibe}")
        
        # Build the graph
        transition_graph = {}
        
        for from_segment in all_segments:
            # Initialize empty list for this segment
            segment_id = from_segment['id']
            transition_graph[segment_id] = []
            
            # Get potential transitions to segments in other tracks
            potential_transitions = []
            
            for to_segment in all_segments:
                # Skip same track segments
                if self._is_same_track(from_segment, to_segment):
                    continue
                
                # Even if not compatible, still give it a small chance
                random_chance = random.random() < self.randomness_factor * 0.5
                
                # Check if transition is possible and get compatibility score
                is_compatible, score = self._vibe_check_compatibility(from_segment, to_segment, current_vibe)
                
                if is_compatible or random_chance:
                    # If it's a random chance transition, assign a lower but non-zero score
                    if not is_compatible and random_chance:
                        score = 0.1 + random.random() * 0.3  # Random score between 0.1-0.4
                    
                    # Add some unpredictability
                    creative_score = self._add_creative_flair(from_segment, to_segment)
                    
                    # Adjust weights based on randomness factor - more creative when randomness is high
                    technical_weight = max(0.3, 0.7 - self.randomness_factor)
                    creative_weight = min(0.7, 0.3 + self.randomness_factor)
                    
                    final_score = technical_weight * score + creative_weight * creative_score
                    
                    # Sometimes just give a completely random boost
                    if random.random() < SURPRISE_TRANSITION_CHANCE:
                        final_score += random.random() * 0.4
                        
                    # Cap at 1.0
                    final_score = min(1.0, final_score)
                    
                    transition_types = self._get_transition_types(from_segment, to_segment)
                    
                    transition = {
                        'to_segment': to_segment['id'],
                        'compatibility': final_score,
                        'transition_types': transition_types,
                        'creative_score': creative_score
                    }
                    potential_transitions.append(transition)
            
            # If no transitions found, create at least one random transition
            if not potential_transitions and all_segments:
                # Find a random segment from another track
                other_tracks = [s for s in all_segments if not self._is_same_track(from_segment, s)]
                if other_tracks:
                    random_segment = random.choice(other_tracks)
                    transition_types = self._get_transition_types(from_segment, random_segment)
                    transition = {
                        'to_segment': random_segment['id'],
                        'compatibility': 0.3,  # Low but non-zero compatibility
                        'transition_types': transition_types,
                        'creative_score': 0.5
                    }
                    potential_transitions.append(transition)
                    self.logger.info(f"Created a random transition for segment {segment_id} as fallback")
            
            # Sort transitions by compatibility score
            potential_transitions.sort(key=lambda t: t['compatibility'], reverse=True)
            
            # Always pick a completely random transition for more unpredictability
            if len(potential_transitions) > 4:
                random_idx = random.randint(0, len(potential_transitions) - 1)
                potential_transitions.insert(1, potential_transitions.pop(random_idx))
            
            # Sometimes completely shuffle the top few for maximum unpredictability
            if random.random() < self.randomness_factor:
                top_n = min(4, len(potential_transitions))
                if top_n > 1:
                    top_part = potential_transitions[:top_n]
                    random.shuffle(top_part)
                    potential_transitions[:top_n] = top_part
                    self.logger.info("Shuffled top transitions for extra randomness!")
            
            # Take top N transitions - but sometimes take more or fewer
            n_transitions = 5
            if random.random() < 0.3:
                n_transitions = random.randint(3, 8)
            
            transition_graph[segment_id] = potential_transitions[:min(n_transitions, len(potential_transitions))]
        
        self.transition_graph = transition_graph
        return transition_graph
    
    def _check_basic_compatibility(self, from_segment: Dict[str, Any], to_segment: Dict[str, Any]) -> Tuple[bool, bool, bool]:
        """
        Check basic musical compatibility between segments
        
        Args:
            from_segment: Source segment
            to_segment: Target segment
            
        Returns:
            Tuple of (bpm_compatible, key_compatible, energy_compatible)
        """
        # Get BPM values
        from_bpm = from_segment.get('bpm', 120)
        to_bpm = to_segment.get('bpm', 120)
        
        # Check if BPMs are compatible
        bpm_ratio = max(from_bpm, to_bpm) / min(from_bpm, to_bpm)
        bpm_difference = abs(from_bpm - to_bpm)
        
        # BPMs are compatible if they're close enough or have an integer ratio
        # But sometimes we'll allow larger differences for creative transitions
        if bpm_difference <= MAX_BPM_DIFFERENCE:
            bpm_compatible = True
        elif 0.99 <= bpm_ratio / round(bpm_ratio) <= 1.01:  # Close to integer ratio
            bpm_compatible = True
        else:
            # Even if not technically compatible, sometimes allow it anyway
            bpm_compatible = random.random() < self.randomness_factor * 0.7
        
        # Get key values
        from_key = from_segment.get('key', 'C major')
        to_key = to_segment.get('key', 'C major')
        
        # Check if keys are compatible using the key compatibility chart
        # But sometimes allow incompatible keys for creative tension
        compatible_keys = KEY_COMPATIBILITY.get(from_key, [])
        key_compatible = to_key in compatible_keys
        
        # Even if not compatible, sometimes allow it for creative reasons
        if not key_compatible and random.random() < self.randomness_factor:
            key_compatible = True
            
        # Check energy compatibility
        from_energy = from_segment.get('avg_energy', 0.5)
        to_energy = to_segment.get('avg_energy', 0.5)
        
        # Energy difference threshold - but we sometimes want big changes!
        energy_compatible = abs(from_energy - to_energy) < 0.4
        
        # Even if energy difference is large, sometimes we want that
        if not energy_compatible and random.random() < 0.5:
            energy_compatible = True
            
        return bpm_compatible, key_compatible, energy_compatible
    
    def _vibe_check_compatibility(self, from_segment: Dict[str, Any], to_segment: Dict[str, Any], vibe_pattern: str) -> Tuple[bool, float]:
        """
        Determine if two segments are compatible for a transition with "vibe check"
        instead of just technical compatibility
        
        Args:
            from_segment: Source segment
            to_segment: Target segment
            vibe_pattern: The overall vibe pattern for this mashup
            
        Returns:
            Tuple of (is_compatible, compatibility_score)
        """
        # Get basic musical compatibility
        bpm_compatible, key_compatible, energy_compatible = self._check_basic_compatibility(from_segment, to_segment)
        
        # If absolutely nothing is compatible, still give it a small chance to be selected
        if not (bpm_compatible or key_compatible or energy_compatible):
            if random.random() < self.randomness_factor * 0.3:
                return True, 0.2  # Technically compatible but low score
            return False, 0.0
        
        # Get segment types
        from_type = from_segment.get('type', 'unknown')
        to_type = to_segment.get('type', 'unknown')
        
        # Calculate base score from technical compatibility - but with more randomness
        base_score = 0.0
        if bpm_compatible: base_score += 0.3 * (0.8 + random.random() * 0.4)  # 0.24-0.42
        if key_compatible: base_score += 0.3 * (0.8 + random.random() * 0.4)  # 0.24-0.42
        if energy_compatible: base_score += 0.2 * (0.8 + random.random() * 0.4)  # 0.16-0.28
        
        # Add "vibe factor" based on segment types and position
        vibe_factor = 0.0
        
        # Create interesting and diverse transitions
        if from_type == 'chorus' and to_type == 'chorus':
            # Direct chorus-to-chorus can be intense but works for energy
            vibe_factor += random.uniform(0.3, 0.8) if random.random() < 0.6 else random.uniform(-0.3, 0.2)
        
        elif from_type == 'verse' and to_type == 'chorus':
            # Classic buildup
            vibe_factor += random.uniform(0.5, 0.9)
        
        elif from_type == 'chorus' and to_type == 'verse':
            # Energy drop - sometimes good for dynamic changes
            vibe_factor += random.uniform(0.2, 0.8)
        
        elif from_type == 'intro' and to_type != 'outro':
            # Good to start with intros
            vibe_factor += random.uniform(0.4, 0.8)
            
        elif from_type == 'bridge' and (to_type == 'chorus' or to_type == 'verse'):
            # Bridge to chorus/verse is classic
            vibe_factor += random.uniform(0.5, 0.9)
            
        # Add energy flow factor based on the vibe pattern
        from_energy = from_segment.get('avg_energy', 0.5)
        to_energy = to_segment.get('avg_energy', 0.5)
        
        # How much should we follow the vibe pattern vs just go random?
        if random.random() < VIBE_PATTERN_WEIGHT:
            # Determine if this energy transition matches our desired vibe pattern
            pattern_values = self.vibe_patterns[vibe_pattern]
            
            # Find where we are in the pattern
            energy_delta = to_energy - from_energy
            
            # Check if this energy change fits our pattern
            for i in range(len(pattern_values) - 1):
                pattern_delta = pattern_values[i+1] - pattern_values[i]
                if (energy_delta > 0 and pattern_delta > 0) or (energy_delta < 0 and pattern_delta < 0):
                    vibe_factor += random.uniform(0.2, 0.4)
                    break
        else:
            # Just go with random vibes
            vibe_factor += random.uniform(-0.2, 0.5)
        
        # Add randomness for unpredictability and creative transitions
        vibe_factor += random.uniform(-self.randomness_factor, self.randomness_factor)
        
        # Calculate final score
        final_score = base_score + vibe_factor
        final_score = max(0.0, min(1.0, final_score))
        
        # We're much more permissive about compatibility now
        return True, final_score
    
    def _add_creative_flair(self, from_segment: Dict[str, Any], to_segment: Dict[str, Any]) -> float:
        """Add creative scoring for unexpected but interesting transitions"""
        
        score = 0.5  # Start neutral
        
        # Contrasting transitions can be interesting
        from_energy = from_segment.get('avg_energy', 0.5)
        to_energy = to_segment.get('avg_energy', 0.5)
        
        # Big energy contrasts can be exciting
        energy_diff = abs(to_energy - from_energy)
        if energy_diff > 0.4:
            score += 0.3 * random.random()  # Sometimes reward dramatic changes
        
        # Sometimes slower to faster transitions feel good
        from_bpm = from_segment.get('bpm', 120)
        to_bpm = to_segment.get('bpm', 120)
        if to_bpm > from_bpm + 5:
            score += 0.2
            
        # Sometimes key changes are interesting
        from_key = from_segment.get('key', 'C')
        to_key = to_segment.get('key', 'C')
        if from_key != to_key:
            score += 0.15 * random.random()
            
        # Add some pure randomness for unexpected transitions
        score += random.uniform(-0.2, 0.3)
        
        return max(0.0, min(1.0, score))
    
    def _get_transition_types(self, from_segment: Dict[str, Any], to_segment: Dict[str, Any]) -> List[str]:
        """Determine appropriate transition types between segments"""
        
        transition_types = []
        
        # Get basic properties
        from_type = from_segment.get('type', 'unknown')
        to_type = to_segment.get('type', 'unknown')
        from_energy = from_segment.get('avg_energy', 0.5)
        to_energy = to_segment.get('avg_energy', 0.5)
        energy_diff = to_energy - from_energy
        
        # Always include at least one basic transition
        basic_transitions = ['cut', 'crossfade', 'beatmatch_crossfade']
        transition_types.append(random.choice(basic_transitions))
        
        # Energy based transitions
        if energy_diff > 0.2:
            energy_up_transitions = ['filter_sweep', 'spinup', 'drum_fill', 'reverse_cymbal']
            transition_types.append(random.choice(energy_up_transitions))
        elif energy_diff < -0.2:
            energy_down_transitions = ['echo_out', 'tape_stop', 'filter_sweep', 'power_down']
            transition_types.append(random.choice(energy_down_transitions))
            
        # Add creative transitions regardless of energy
        creative_transitions = [
            'scratch_in', 'glitch_effect', 'stutter_cut', 'vocal_sample', 
            'beat_roll', 'bass_drop', 'dj_shout'
        ]
        
        # Add between 1-3 creative transition types
        num_creative = random.randint(1, 3)
        for _ in range(num_creative):
            if creative_transitions:
                selected = random.choice(creative_transitions)
                transition_types.append(selected)
                creative_transitions.remove(selected)  # Don't repeat
        
        # Sometimes completely randomize all transition types for max unpredictability
        if random.random() < self.randomness_factor * 0.5:
            transition_types = random.sample(TRANSITION_TYPES, min(4, len(TRANSITION_TYPES)))
            
        return transition_types

    def _is_same_track(self, segment1: Dict[str, Any], segment2: Dict[str, Any]) -> bool:
        """Check if two segments are from the same track"""
        # Simple string comparison of track name parts of the ID
        track1 = segment1['id'].split('_')[0] if 'id' in segment1 else segment1.get('track', 'unknown')
        track2 = segment2['id'].split('_')[0] if 'id' in segment2 else segment2.get('track', 'unknown')
        
        # Try matching with track attribute if available
        if 'track' in segment1 and 'track' in segment2:
            return segment1['track'] == segment2['track']
            
        return track1 == track2


def build_transition_graph(tracks_data: Dict[str, Dict[str, Any]], randomness_boost=1.0) -> Dict:
    """
    Build a transition graph from track data
    
    Args:
        tracks_data: Dictionary mapping track names to their features and segments
        randomness_boost: Factor to increase randomness/creativity (default: 1.0)
        
    Returns:
        Transition graph dictionary
    """
    logger.info("Building transition graph")
    
    # Extract all segments by track
    segments_by_track = {}
    for track_name, track_data in tracks_data.items():
        if 'segments' in track_data:
            segments_by_track[track_name] = track_data['segments']
            logger.info(f"Found {len(track_data['segments'])} segments for track {track_name}")
    
    # Debug output to verify what tracks and segments we have
    for track_name, segments in segments_by_track.items():
        for segment in segments:
            logger.info(f"Track: {track_name}, Segment: {segment['id']}, Type: {segment.get('type', 'unknown')}")
    
    # Build the transition matrix
    engine = TransitionEngine(randomness_boost=randomness_boost)
    transition_graph = engine.build_transition_graph(segments_by_track)
    
    # Convert the graph to a dictionary format for easier consumption
    transition_matrix = {}
    
    for from_id, transitions in transition_graph.items():
        transition_matrix[from_id] = {}
        
        for transition in transitions:
            to_id = transition['to_segment']
            
            # Get the best transition type
            transition_types = transition.get('transition_types', ['crossfade'])
            
            # With higher randomness, sometimes pick a random transition type instead of the first one
            if random.random() < 0.4 * randomness_boost and len(transition_types) > 1:
                transition_type = random.choice(transition_types)
            else:
                transition_type = transition_types[0] if transition_types else DEFAULT_TRANSITION_TYPE
            
            # Calculate transition duration based on type
            if transition_type in ['cut', 'scratch_in']:
                duration = 0.5
            elif transition_type in ['beatmatch_crossfade', 'harmonic_crossfade']:
                duration = DEFAULT_TRANSITION_DURATION_SECONDS
            elif transition_type in ['echo_out', 'filter_sweep']:
                duration = DEFAULT_TRANSITION_DURATION_SECONDS * 1.5
            else:
                duration = DEFAULT_TRANSITION_DURATION_SECONDS
                
            # Add some randomness to duration with higher randomness_boost
            if random.random() < 0.3 * randomness_boost:
                duration *= 0.7 + random.random() * 0.6  # 70-130% of original
            
            # Store transition data
            transition_matrix[from_id][to_id] = {
                'score': transition['compatibility'],
                'transition_type': transition_type,
                'duration': duration
            }
            logger.info(f"Created transition from {from_id} to {to_id} with type {transition_type}")
    
    # Verify the transitions that were created
    transition_count = sum(len(transitions) for transitions in transition_matrix.values())
    logger.info(f"Created {transition_count} transitions across {len(transition_matrix)} segments")
    
    # Get track names and segment IDs for each track
    tracks_and_segments = {}
    for segment_id in transition_matrix.keys():
        # Extract track name from segment ID (assuming format like "track_segmentnum")
        parts = segment_id.split('_')
        if len(parts) > 1:
            track_name = parts[0]
            if track_name not in tracks_and_segments:
                tracks_and_segments[track_name] = []
            tracks_and_segments[track_name].append(segment_id)
        else:
            # Try to get track from segment data
            for track_name, segments in segments_by_track.items():
                for segment in segments:
                    if segment['id'] == segment_id:
                        if track_name not in tracks_and_segments:
                            tracks_and_segments[track_name] = []
                        tracks_and_segments[track_name].append(segment_id)
                        break

    # Ensure each track has transitions to every other track
    logger.info(f"Ensuring transitions between all {len(tracks_and_segments)} tracks")
    track_names = list(tracks_and_segments.keys())
    
    # For each track, ensure it has a transition to every other track
    for from_track_idx, from_track in enumerate(track_names):
        from_segments = tracks_and_segments[from_track]
        
        # Get best segments to transition from (prefer chorus/verse segments)
        best_from_segments = []
        for segment_id in from_segments:
            for track_name, segments in segments_by_track.items():
                if track_name == from_track:
                    for segment in segments:
                        if segment['id'] == segment_id:
                            segment_type = segment.get('type', 'unknown')
                            if segment_type in ['chorus', 'verse', 'bridge']:
                                best_from_segments.append(segment_id)
        
        # If no good segments found, use any segments
        if not best_from_segments:
            best_from_segments = from_segments
            
        # For each other track, check/create transitions
        for to_track_idx, to_track in enumerate(track_names):
            if from_track == to_track:
                continue  # Skip same track
                
            to_segments = tracks_and_segments[to_track]
            
            # Get best segments to transition to (prefer intro/verse segments)
            best_to_segments = []
            for segment_id in to_segments:
                for track_name, segments in segments_by_track.items():
                    if track_name == to_track:
                        for segment in segments:
                            if segment['id'] == segment_id:
                                segment_type = segment.get('type', 'unknown')
                                if segment_type in ['intro', 'verse', 'chorus']:
                                    best_to_segments.append(segment_id)
            
            # If no good segments found, use any segments
            if not best_to_segments:
                best_to_segments = to_segments
                
            # Check if any transitions already exist from from_track to to_track
            has_transition = False
            for from_segment in from_segments:
                for to_segment in to_segments:
                    if from_segment in transition_matrix and to_segment in transition_matrix[from_segment]:
                        has_transition = True
                        break
                if has_transition:
                    break
            
            # If no transition exists, create one between best segments
            if not has_transition and best_from_segments and best_to_segments:
                from_segment = random.choice(best_from_segments)
                to_segment = random.choice(best_to_segments)
                
                # Ensure from_segment has an entry in transition_matrix
                if from_segment not in transition_matrix:
                    transition_matrix[from_segment] = {}
                    
                # Create the forced transition with reasonable defaults
                transition_matrix[from_segment][to_segment] = {
                    'score': 0.7,
                    'transition_type': 'crossfade',
                    'duration': DEFAULT_TRANSITION_DURATION_SECONDS
                }
                
                logger.info(f"Added forced transition from track {from_track} to {to_track} ({from_segment} -> {to_segment})")
    
    # Final verification
    transition_count = sum(len(transitions) for transitions in transition_matrix.values())
    logger.info(f"Final transition matrix has {transition_count} transitions across {len(transition_matrix)} segments")
    
    # Verify track-to-track connectivity
    track_connectivity = {track: set() for track in track_names}
    
    for from_segment, transitions in transition_matrix.items():
        from_track = None
        for track, segments in tracks_and_segments.items():
            if from_segment in segments:
                from_track = track
                break
                
        if from_track:
            for to_segment in transitions.keys():
                to_track = None
                for track, segments in tracks_and_segments.items():
                    if to_segment in segments:
                        to_track = track
                        break
                        
                if to_track and to_track != from_track:
                    track_connectivity[from_track].add(to_track)
    
    # Log connectivity
    for track, connected_tracks in track_connectivity.items():
        logger.info(f"Track {track} has transitions to {len(connected_tracks)} other tracks: {', '.join(connected_tracks)}")
    
    return transition_matrix


if __name__ == "__main__":
    # Test code
    import json
    import sys
    from pathlib import Path
    
    # Load sample data from files if available
    if len(sys.argv) > 1:
        data_dir = Path(sys.argv[1])
        tracks_data = {}
        
        # Load features and segments from JSON files
        for file_path in data_dir.glob("*.json"):
            with open(file_path, 'r') as f:
                tracks_data[file_path.stem] = json.load(f)
        
        if tracks_data:
            engine = TransitionEngine()
            transition_graph = engine.build_transition_graph(tracks_data)
            
            # Print some transition examples
            for source_id, targets in list(transition_graph.items())[:5]:
                print(f"\nTransitions from {source_id}:")
                for target_id, info in list(targets.items())[:3]:
                    print(f"  → {target_id}: score={info['compatibility']:.2f}, types={', '.join(info['transition_types'])}")
        else:
            print("No data files found.")
    else:
        print("Please provide a directory with track data JSON files.") 