import os
import numpy as np
import networkx as nx
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import traceback
import json

from transitions_ai.src.config import (
    MIN_SONGS_IN_MASHUP, MAX_CONSECUTIVE_PHRASES_FROM_SAME_SONG,
    MIN_PHRASES_PER_SONG, QUALITY_THRESHOLD, MAX_MASHUP_LENGTH_MINUTES,
    PATH_FINDING_HEURISTIC_WEIGHT, MAX_BACKTRACKING_ATTEMPTS,
    VISUALIZATION_DIR, RANDOMNESS_FACTOR, SURPRISE_TRANSITION_CHANCE
)
from transitions_ai.src.logger import get_logger, log_mashup_generation

logger = get_logger("mashup")

class MashupGenerator:
    """
    Generates DJ mashups by finding optimal paths through the transition graph
    """
    
    def __init__(self, tracks_data: Dict[str, Dict[str, Any]], transition_matrix: Dict[str, Dict[str, Dict[str, Any]]]):
        """
        Initialize the mashup generator
        
        Args:
            tracks_data: Dictionary mapping track names to their features and segments
            transition_matrix: Transition compatibility matrix between segments
        """
        self.tracks_data = tracks_data
        self.transition_matrix = transition_matrix
        self.graph = None
        logger.info(f"Initializing mashup generator with {len(tracks_data)} tracks and {sum(len(v) for v in transition_matrix.values())} transitions")
    
    def build_graph(self) -> nx.DiGraph:
        """
        Build a directed graph from the transition matrix for path finding
        
        Returns:
            NetworkX DiGraph
        """
        logger.info("Building transition graph")
        
        # Create a directed graph
        G = nx.DiGraph()
        
        # Add all segments as nodes
        for track_name, track_data in self.tracks_data.items():
            if 'segments' not in track_data:
                continue
                
            for segment in track_data['segments']:
                segment_id = segment['id']
                # Add node with segment data
                G.add_node(segment_id, 
                          track=track_name, 
                          type=segment['type'],
                          start=segment['start'],
                          end=segment['end'],
                          duration=segment['duration'])
        
        # Add transitions as edges
        for source_id, targets in self.transition_matrix.items():
            for target_id, transition in targets.items():
                # Add edge with transition data
                G.add_edge(source_id, target_id, 
                          weight=1.0 - transition['score'],  # Convert score to weight (lower is better)
                          score=transition['score'],
                          type=transition['transition_type'],
                          duration=transition['duration'])
        
        self.graph = G
        logger.info(f"Built graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G
    
    def generate_mashup(self, target_duration_minutes: Optional[float] = None, 
                       start_track: Optional[str] = None,
                       end_track: Optional[str] = None,
                       min_tracks: int = MIN_SONGS_IN_MASHUP,
                       max_consecutive: int = MAX_CONSECUTIVE_PHRASES_FROM_SAME_SONG,
                       randomness_boost: float = 1.0) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Generate an optimal mashup by finding the best path through the graph
        
        Args:
            target_duration_minutes: Target duration in minutes (default: config value)
            start_track: Track to start with (default: random)
            end_track: Track to end with (default: random)
            min_tracks: Minimum number of tracks to include
            max_consecutive: Maximum consecutive segments from the same track
            randomness_boost: Factor to increase randomness/creativity (default: 1.0)
            
        Returns:
            Tuple of (segments list, transitions list)
        """
        logger.info("Generating mashup")
        logger.info(f"Randomness boost: {randomness_boost}")
        
        # Adjusted randomness based on boost
        randomness = min(0.9, random.uniform(0.1, 0.8) * randomness_boost)
        
        # Build graph if not already built
        if self.graph is None:
            self.build_graph()
        
        # Check if we have enough nodes in the graph
        if self.graph.number_of_nodes() == 0:
            logger.error("No segments found in graph")
            return [], []
            
        # Check if we have any transitions
        if self.graph.number_of_edges() == 0:
            logger.warning("No transitions found in graph. Creating a basic mashup")
            return self._create_simple_mashup(self.tracks_data, target_duration_minutes * 60)
        
        # Set target duration
        if target_duration_minutes is None:
            target_duration_minutes = MAX_MASHUP_LENGTH_MINUTES
        target_duration_seconds = target_duration_minutes * 60
        
        # Decide on a mashup vibe strategy
        vibe_strategies = [
            "energy_flow", "surprising_contrasts", "genre_fusion", 
            "mood_journey", "tempo_ride", "pure_chaos"
        ]
        chosen_vibe = random.choice(vibe_strategies)
        logger.info(f"Chosen mashup vibe strategy: {chosen_vibe}")
        
        # Get all nodes (segments)
        segments = list(self.graph.nodes(data=True))
        
        # Shuffle list of segments for random starting points
        random.shuffle(segments)
        
        # Filter start and end segments if specified
        start_segments = [s for s, data in segments if start_track is None or data['track'] == start_track]
        end_segments = [s for s, data in segments if end_track is None or data['track'] == end_track]
        
        if not start_segments:
            logger.warning(f"No valid start segments found. Using any segment.")
            start_segments = [s for s, _ in segments]
            
        if not end_segments:
            logger.warning(f"No valid end segments found. Using any segment.")
            end_segments = [s for s, _ in segments]
            
        # Group segments by track to ensure we include multiple tracks
        segments_by_track = {}
        for seg_id, data in segments:
            track_name = data['track']
            if track_name not in segments_by_track:
                segments_by_track[track_name] = []
            segments_by_track[track_name].append(seg_id)
            
        # Check if we have multiple tracks
        if len(segments_by_track) < 2:
            logger.warning("Only one track found. Creating a basic mashup")
            return self._create_simple_mashup(self.tracks_data, target_duration_seconds)
        
        # Choose better starting segments based on vibe
        if chosen_vibe == "energy_flow":
            # Look for intro segments or lower energy segments to start with
            start_segments = sorted(
                [(s, self.graph.nodes[s].get('type', 'unknown'), self.graph.nodes[s].get('energy', 0.5)) 
                 for s in start_segments], 
                key=lambda x: (0 if x[1] == 'intro' else 1, x[2])
            )
            start_segments = [s[0] for s in start_segments[:max(3, len(start_segments))]]
        
        elif chosen_vibe == "pure_chaos":
            # Just pick totally random starting points
            random.shuffle(start_segments)
        
        # Try different starting segments until a valid path is found
        best_paths = []  # Keep multiple paths to choose from
        attempts = 0
        
        for start_segment in start_segments:
            # Skip if we've made too many attempts
            if attempts >= MAX_BACKTRACKING_ATTEMPTS:
                break
                
            attempts += 1
            logger.info(f"Attempt {attempts}: Starting from {start_segment}")
            
            try:
                # Find path using modified A* search with the chosen vibe
                path, score = self._find_path(start_segment, target_duration_seconds, 
                                            end_segments, min_tracks, max_consecutive,
                                            vibe_strategy=chosen_vibe, randomness=randomness)
                
                if path:
                    best_paths.append((path, score))
                    logger.info(f"Found path with score {score:.2f}")
                    
                    # If we found paths, stop searching after a reasonable number
                    if len(best_paths) >= 3:
                        break
            except Exception as e:
                logger.error(f"Error finding path: {str(e)}")
                logger.error(traceback.format_exc())
        
        if not best_paths:
            logger.warning("Failed to find valid mashup path")
            return [], []
        
        # Choose the final path - sometimes pick the best one, sometimes pick a random one
        if random.random() < 0.7:
            # Pick the best path
            best_paths.sort(key=lambda x: x[1], reverse=True)
            chosen_path = best_paths[0][0]
        else:
            # Pick a random path for more surprise
            chosen_path = random.choice(best_paths)[0]
            logger.info("Chose a random path instead of the highest scoring one for more surprise!")
        
        # Convert path to segment and transition lists
        segments, transitions = self._path_to_mashup(chosen_path)
        
        # Sometimes deliberately rearrange parts of the mashup for creative effect
        if random.random() < 0.3 and len(segments) > 6:
            logger.info("Applying creative rearrangement to the mashup!")
            # Select a segment in the middle to move around
            start_idx = random.randint(1, len(segments) // 3)
            move_idx = random.randint(start_idx + 2, len(segments) - 2)
            insert_idx = random.randint(1, len(segments) - 2)
            
            # Move the segment
            segment_to_move = segments.pop(move_idx)
            segments.insert(insert_idx, segment_to_move)
            
            # Transitions need to be recalculated
            transitions = self._recalculate_transitions(segments)
        
        # Calculate total duration
        total_duration = sum(segment['duration'] for segment in segments)
        num_transitions = len(transitions)
        
        # Log mashup generation
        log_mashup_generation(segments, total_duration, num_transitions)
        
        logger.info(f"Generated mashup with {len(segments)} segments, {num_transitions} transitions, duration: {total_duration:.2f}s")
        return segments, transitions
    
    def _find_path(self, start_segment: str, target_duration: float, 
                  end_segments: List[str], min_tracks: int, max_consecutive: int,
                  vibe_strategy: str = "energy_flow", randomness: float = 0.3) -> Tuple[List[str], float]:
        """
        Find a path through the graph using a modified A* search algorithm
        
        Args:
            start_segment: Starting segment ID
            target_duration: Target duration in seconds
            end_segments: List of valid ending segment IDs
            min_tracks: Minimum number of tracks to include
            max_consecutive: Maximum consecutive segments from the same track
            vibe_strategy: Strategy for selecting transitions
            randomness: How much to prioritize unexpected combinations
            
        Returns:
            Tuple of (path as list of segment IDs, path score)
        """
        # Check if we have outgoing edges from the start segment
        if len(list(self.graph.successors(start_segment))) == 0:
            logger.warning(f"No outgoing transitions from {start_segment}. Creating a basic path.")
            return self._create_basic_path(start_segment, target_duration)
            
        # A* search to find optimal path
        # The heuristic is based on how close we are to the target duration
        
        # Priority queue (segment_id, path_so_far, duration_so_far, tracks_used, current_track, consecutive_count, total_score)
        queue = [(start_segment, [start_segment], self.graph.nodes[start_segment]['duration'], 
                 {self.graph.nodes[start_segment]['track']}, self.graph.nodes[start_segment]['track'], 1, 0)]
        
        # Track visited states to avoid cycles
        visited = set()
        
        # Energy and type sequences for different vibes
        energy_targets = {
            "energy_flow": [0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 0.7, 0.5],
            "surprising_contrasts": [0.5, 0.8, 0.3, 0.9, 0.4, 0.7],
            "mood_journey": [0.4, 0.3, 0.5, 0.7, 0.8, 0.7, 0.4],
            "tempo_ride": [0.5, 0.6, 0.7, 0.8, 0.7, 0.5, 0.6]
        }
        
        # Keep track of all paths we've tried
        all_paths = []
        best_partial_path = None
        best_partial_score = 0
        best_partial_duration = 0
        best_partial_tracks_count = 0
        
        # Number of iterations to allow before giving up
        max_iterations = 1000
        iterations = 0
        
        while queue and iterations < max_iterations:
            iterations += 1
            
            # Occasionally introduce randomness in the queue ordering
            if random.random() < randomness:
                # Shuffle or pick a random element from the top half
                if len(queue) > 2:
                    idx = random.randint(0, min(5, len(queue) - 1))
                    temp = queue[0]
                    queue[0] = queue[idx]
                    queue[idx] = temp
                    logger.debug("Added randomness to path selection")
            else:
                # Sort queue by combined score (A* heuristic, but modified by vibe strategy)
                queue.sort(key=lambda x: (
                    # Heuristic: how close we are to target duration - less important now
                    (PATH_FINDING_HEURISTIC_WEIGHT * 0.7) * abs(target_duration - x[2]) +
                    # Edge weights (lower is better)
                    -x[6] * (1.0 - randomness * 0.5)  # Reduce importance when randomness is high
                ))
            
            # Get the best candidate
            current, path, duration, tracks_used, current_track, consecutive, score = queue.pop(0)
            
            # Keep track of best partial path for fallback
            if (len(tracks_used) > best_partial_tracks_count or 
                (len(tracks_used) == best_partial_tracks_count and 
                 abs(duration - target_duration) < abs(best_partial_duration - target_duration))):
                best_partial_path = path
                best_partial_score = score
                best_partial_duration = duration
                best_partial_tracks_count = len(tracks_used)
            
            # Skip if we've visited this state
            state = (current, tuple(sorted(tracks_used)), current_track, consecutive)
            if state in visited:
                continue
            visited.add(state)
            
            # Track all paths we've tried
            all_paths.append((path, score, duration, len(tracks_used)))
            
            # Check if we're close enough to target duration and we've used minimum tracks
            # Much more flexible requirements now
            duration_ratio = abs(duration - target_duration) / target_duration
            if ((duration_ratio < 0.25 and len(tracks_used) >= min(min_tracks, 2)) or  # Very flexible requirement
                (duration > target_duration * 0.8 and len(path) >= 3) or  # Accept shorter paths
                (current in end_segments and duration > target_duration * 0.7)):  # Accept shorter paths if they end properly
                return path, score / len(path)
            
            # If we've exceeded max duration, skip
            if duration > target_duration * 1.5:  # More flexible max duration
                continue
                
            # Get outgoing edges (transitions)
            outgoing = list(self.graph.out_edges(current, data=True))
            
            # If no outgoing edges but we have a decent path, return it
            if not outgoing and len(path) >= 3 and len(tracks_used) >= 2:
                logger.info(f"No more transitions from {current}, but we have a decent path with {len(path)} segments.")
                return path, score / len(path)
            
            # If no outgoing edges, skip this state
            if not outgoing:
                continue
                
            # Apply vibe strategy to sort outgoing edges
            if vibe_strategy == "energy_flow" and outgoing:
                # Get target energy for this position in the mashup
                progress = min(1.0, duration / target_duration)
                target_idx = int(progress * len(energy_targets.get(vibe_strategy, [0.5])))
                target_energy = energy_targets.get(vibe_strategy, [0.5])[target_idx % len(energy_targets.get(vibe_strategy, [0.5]))]
                
                # Sort by how close the next segment's energy is to our target
                outgoing.sort(key=lambda x: (
                    abs(self.graph.nodes[x[1]].get('energy', 0.5) - target_energy),
                    x[2]['weight']
                ))
                
            elif vibe_strategy == "surprising_contrasts" and outgoing:
                # Sort to prefer energy contrasts
                current_energy = self.graph.nodes[current].get('energy', 0.5)
                outgoing.sort(key=lambda x: (
                    -abs(self.graph.nodes[x[1]].get('energy', 0.5) - current_energy),  # Bigger difference is better
                    x[2]['weight']
                ))
                
            elif vibe_strategy == "pure_chaos" and outgoing:
                # Just randomize the order completely
                random.shuffle(outgoing)
                
            else:
                # Default sorting by edge weight (score)
                outgoing.sort(key=lambda x: x[2]['weight'])
            
            # Add randomness to the selection
            if outgoing and len(outgoing) > 2 and random.random() < randomness:
                # Randomly promote a non-top transition
                idx = random.randint(1, len(outgoing) - 1)
                outgoing[0], outgoing[idx] = outgoing[idx], outgoing[0]
            
            # Examine each potential next segment (limit number of successors to consider)
            max_successors = min(8, len(outgoing))
            for _, next_segment, edge_data in outgoing[:max_successors]:
                # Skip if already in path (avoid cycles) - but sometimes allow revisiting tracks in long paths
                if next_segment in path and (len(path) < 10 or random.random() > randomness * 0.3):
                    continue
                
                next_track = self.graph.nodes[next_segment]['track']
                next_duration = duration + self.graph.nodes[next_segment]['duration'] + edge_data.get('duration', 0)
                
                # Check if we would exceed max consecutive segments from same track
                new_consecutive = consecutive + 1 if next_track == current_track else 1
                
                # Sometimes be more lenient with consecutive segments constraint
                if new_consecutive > max_consecutive and random.random() > randomness * 0.4:
                    continue
                
                # Add to tracks used
                new_tracks_used = tracks_used.copy()
                new_tracks_used.add(next_track)
                
                # Calculate new score - adjust based on vibe strategy
                edge_score = edge_data.get('score', 0.5)  # Get score or default to 0.5
                if 'weight' in edge_data:
                    edge_score = 1.0 - edge_data['weight']  # Convert weight back to score (0-1)
                
                # Apply vibe strategy modifiers
                if vibe_strategy == "energy_flow":
                    # Reward following the energy flow pattern
                    target_idx = min(len(path), len(energy_targets.get(vibe_strategy, [])) - 1)
                    target_energy = energy_targets.get(vibe_strategy, [0.5])[target_idx]
                    energy_match = 1.0 - abs(self.graph.nodes[next_segment].get('energy', 0.5) - target_energy)
                    edge_score = 0.7 * edge_score + 0.3 * energy_match
                    
                elif vibe_strategy == "surprising_contrasts":
                    # Reward contrasting transitions
                    current_energy = self.graph.nodes[current].get('energy', 0.5)
                    next_energy = self.graph.nodes[next_segment].get('energy', 0.5)
                    contrast = abs(next_energy - current_energy)
                    edge_score = 0.6 * edge_score + 0.4 * contrast
                
                elif vibe_strategy == "pure_chaos":
                    # Add a random factor to the score
                    edge_score = 0.5 * edge_score + 0.5 * random.random()
                
                # Add to queue
                new_score = score + edge_score
                queue.append((next_segment, path + [next_segment], next_duration, 
                             new_tracks_used, next_track, new_consecutive, new_score))
        
        # If we couldn't find a path, check if we have a decent partial path
        if best_partial_path and len(best_partial_path) >= 3:
            logger.warning(f"Couldn't find perfect path, using best partial path with {len(best_partial_path)} segments")
            return best_partial_path, best_partial_score / len(best_partial_path)
        
        # If still no path, create a basic path
        logger.warning(f"Couldn't find any valid path from {start_segment}, creating basic path")
        return self._create_basic_path(start_segment, target_duration)
    
    def _create_basic_path(self, start_segment: str, target_duration: float) -> Tuple[List[str], float]:
        """Create a simple path when pathfinding fails"""
        logger.info("Creating a basic path using available segments")
        
        path = [start_segment]
        duration = self.graph.nodes[start_segment]['duration']
        start_track = self.graph.nodes[start_segment]['track']
        tracks_used = {start_track}
        
        # Get all segments
        all_segments = list(self.graph.nodes())
        
        # Group segments by track
        segments_by_track = {}
        for segment_id in all_segments:
            track = self.graph.nodes[segment_id]['track']
            if track not in segments_by_track:
                segments_by_track[track] = []
            segments_by_track[track].append(segment_id)
        
        # If we have multiple tracks, ensure we use them
        tracks = list(segments_by_track.keys())
        if len(tracks) > 1:
            current_track_idx = tracks.index(start_track) if start_track in tracks else 0
            
            # Try to add a segment from each track in sequence
            while duration < target_duration * 0.8 and len(path) < 12:
                # Move to next track
                current_track_idx = (current_track_idx + 1) % len(tracks)
                next_track = tracks[current_track_idx]
                
                # Skip if same as current track
                if next_track == tracks[tracks.index(self.graph.nodes[path[-1]]['track'])]:
                    continue
                
                # Get segments for this track
                track_segments = segments_by_track[next_track]
                if not track_segments:
                    continue
                    
                # Pick a segment that's not already in path
                available_segments = [s for s in track_segments if s not in path]
                if not available_segments:
                    continue
                    
                # Pick first segment if "intro" is available, otherwise random
                intro_segments = [s for s in available_segments if self.graph.nodes[s].get('type', '') == 'intro']
                if intro_segments:
                    next_segment = intro_segments[0]
                else:
                    next_segment = random.choice(available_segments)
                
                # Add to path
                path.append(next_segment)
                duration += self.graph.nodes[next_segment]['duration'] + 2.0  # Add transition duration
                tracks_used.add(next_track)
        else:
            # Only one track, add some segments from it
            track_segments = segments_by_track.get(start_track, [])
            
            # Sort by start time
            track_segments.sort(key=lambda s: self.graph.nodes[s].get('start', 0))
            
            # Add segments in sequence
            for segment_id in track_segments:
                if segment_id != start_segment and len(path) < 5:
                    path.append(segment_id)
                    duration += self.graph.nodes[segment_id]['duration']
        
        # Calculate a reasonable score
        score = 0.5 * len(path)
        
        return path, score / len(path)
    
    def _create_simple_mashup(self, tracks_data: Dict[str, Dict[str, Any]], target_duration: float) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Create a simple mashup when transition graph approach fails
        
        Args:
            tracks_data: Dictionary of track data
            target_duration: Target duration in seconds
            
        Returns:
            Tuple of (segments, transitions)
        """
        logger.info("Creating a simple mashup from available tracks")
        
        segments = []
        transitions = []
        
        # Collect all tracks with segments
        viable_tracks = {}
        for track_name, track_data in tracks_data.items():
            if 'segments' in track_data and track_data['segments']:
                viable_tracks[track_name] = track_data['segments']
        
        # If no viable tracks, return empty result
        if not viable_tracks:
            logger.error("No viable tracks with segments found")
            return [], []
            
        # Use at least 2 tracks if available
        track_names = list(viable_tracks.keys())
        random.shuffle(track_names)
        track_count = min(len(track_names), 3)  # Use up to 3 tracks
        selected_tracks = track_names[:track_count]
        
        total_duration = 0
        
        # For each selected track, choose segments
        for track_name in selected_tracks:
            track_segments = viable_tracks[track_name]
            
            # If we have enough duration already, break
            if total_duration >= target_duration:
                break
                
            # Choose intro segment if available, otherwise first segment
            intro_segments = [s for s in track_segments if s.get('type', '') == 'intro']
            if intro_segments:
                first_segment = intro_segments[0]
            else:
                # Sort by start time and take first
                track_segments.sort(key=lambda s: s.get('start', 0))
                first_segment = track_segments[0]
                
            # Add the first segment
            segments.append(first_segment)
            total_duration += first_segment['duration']
            
            # If short on duration, add more segments from this track
            remaining_segments = [s for s in track_segments if s != first_segment]
            remaining_segments.sort(key=lambda s: s.get('start', 0))
            
            # Add up to 2 more segments from the track if needed
            for i, segment in enumerate(remaining_segments):
                if i >= 2 or total_duration >= target_duration:
                    break
                    
                # Add this segment
                segments.append(segment)
                total_duration += segment['duration']
                
                # Create a transition
                if len(segments) > 1:
                    transition = {
                        'from_segment': segments[-2]['id'],
                        'to_segment': segments[-1]['id'],
                        'type': 'crossfade',
                        'duration': 2.0,
                        'score': 0.7
                    }
                    transitions.append(transition)
                    total_duration += transition['duration']
        
        # Create transitions between tracks
        for i in range(len(segments) - 1):
            # Skip if same track or already has transition
            if segments[i]['track'] == segments[i+1]['track']:
                continue
                
            # Check if transition already exists
            from_id = segments[i]['id']
            to_id = segments[i+1]['id']
            
            exists = False
            for t in transitions:
                if t['from_segment'] == from_id and t['to_segment'] == to_id:
                    exists = True
                    break
                    
            if not exists:
                transition = {
                    'from_segment': from_id,
                    'to_segment': to_id,
                    'type': 'crossfade',
                    'duration': 3.0,
                    'score': 0.6
                }
                transitions.append(transition)
                total_duration += transition['duration']
        
        logger.info(f"Created simple mashup with {len(segments)} segments from {len(selected_tracks)} tracks")
        return segments, transitions
    
    def _recalculate_transitions(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Recalculate transitions after segment order has been changed"""
        transitions = []
        
        for i in range(len(segments) - 1):
            from_segment_id = segments[i]['id']
            to_segment_id = segments[i+1]['id']
            
            # Check if transition exists in graph
            if self.graph.has_edge(from_segment_id, to_segment_id):
                # Use existing transition data
                edge_data = self.graph.get_edge_data(from_segment_id, to_segment_id)
                transition = {
                    'from_segment': from_segment_id,
                    'to_segment': to_segment_id,
                    'type': edge_data.get('type', 'crossfade'),
                    'duration': edge_data.get('duration', 2.0),
                    'score': edge_data.get('score', 0.5)
                }
            else:
                # Create a creative transition anyway
                transition_types = ['crossfade', 'filter_sweep', 'echo_out', 'cut', 'spinup']
                transition = {
                    'from_segment': from_segment_id,
                    'to_segment': to_segment_id,
                    'type': random.choice(transition_types),
                    'duration': random.uniform(1.0, 3.0),
                    'score': 0.3  # Lower score for these improvised transitions
                }
            
            transitions.append(transition)
            
        return transitions
    
    def _path_to_mashup(self, path: List[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Convert a path of segment IDs to a mashup structure
        
        Args:
            path: List of segment IDs
            
        Returns:
            Tuple of (segments list, transitions list)
        """
        if not path:
            logger.warning("Empty path - cannot convert to mashup")
            return [], []
            
        segments = []
        transitions = []
        
        # Process segments
        for segment_id in path:
            segment_data = self.graph.nodes[segment_id]
            track_name = segment_data['track']
            
            track_data = self.tracks_data.get(track_name, {})
            
            # Find the segment in the track data
            track_segments = track_data.get('segments', [])
            matching_segments = [s for s in track_segments if s['id'] == segment_id]
            
            if matching_segments:
                # Use the segment from the track data
                segment_info = matching_segments[0].copy()
                segment_info['track'] = track_name
            else:
                # If not found, use data from the graph
                segment_info = {
                    'id': segment_id,
                    'track': track_name,
                    'type': segment_data.get('type', 'unknown'),
                    'start': segment_data.get('start', 0),
                    'end': segment_data.get('end', 0),
                    'duration': segment_data.get('duration', 0),
                    'bpm': track_data.get('bpm', 120),
                    'key': track_data.get('key', 'C major')
                }
            
            segments.append(segment_info)
        
        # Process transitions
        for i in range(len(path) - 1):
            from_id = path[i]
            to_id = path[i + 1]
            
            # Check if this transition exists in the transition matrix
            if from_id in self.transition_matrix and to_id in self.transition_matrix[from_id]:
                # Use transition from the matrix
                transition_data = self.transition_matrix[from_id][to_id]
                
                transition = {
                    'from_segment': from_id,
                    'to_segment': to_id,
                    'type': transition_data.get('transition_type', 'crossfade'),
                    'duration': transition_data.get('duration', 2.0),
                    'score': transition_data.get('score', 0.5)
                }
            else:
                # Create a fallback transition
                logger.warning(f"Transition from {from_id} to {to_id} not found in matrix - creating fallback")
                
                # Try to get edge from graph
                if self.graph.has_edge(from_id, to_id):
                    edge_data = self.graph.get_edge_data(from_id, to_id)
                    transition_type = edge_data.get('type', 'crossfade')
                    duration = edge_data.get('duration', 2.0)
                    score = edge_data.get('score', 0.5)
                else:
                    # Create completely fabricated transition
                    transition_types = ['crossfade', 'cut', 'filter_sweep', 'echo_out']
                    transition_type = random.choice(transition_types)
                    duration = random.uniform(1.5, 3.0)
                    score = 0.3
                
                transition = {
                    'from_segment': from_id,
                    'to_segment': to_id,
                    'type': transition_type,
                    'duration': duration,
                    'score': score
                }
            
            transitions.append(transition)
        
        return segments, transitions
    
    def visualize_mashup(self, segments: List[Dict[str, Any]], transitions: List[Dict[str, Any]], 
                        save_path: Optional[Path] = None) -> None:
        """
        Visualize the mashup structure
        
        Args:
            segments: List of segment data
            transitions: List of transition data
            save_path: Path to save the visualization
        """
        if not save_path:
            save_path = VISUALIZATION_DIR / "mashup_structure.png"
            
        logger.info(f"Visualizing mashup structure to {save_path}")
        
        try:
            # Create figure
            plt.figure(figsize=(15, 8))
            
            # Track colors
            tracks = list(set(segment['track'] for segment in segments))
            colors = plt.cm.tab10(np.linspace(0, 1, len(tracks)))
            track_colors = dict(zip(tracks, colors))
            
            # Time positions
            time_position = 0
            for i, segment in enumerate(segments):
                # Get segment data
                track = segment['track']
                segment_type = segment['type']
                duration = segment['duration']
                
                # Add transition time if not the first segment
                if i > 0:
                    time_position += transitions[i-1]['duration']
                
                # Plot segment as a rectangle
                rect = plt.Rectangle((time_position, 0), duration, 0.8, 
                                   color=track_colors[track], alpha=0.7)
                plt.gca().add_patch(rect)
                
                # Add segment label
                plt.text(time_position + duration/2, 0.4, f"{segment_type}", 
                        ha='center', va='center', color='black', fontweight='bold')
                
                # Add track name
                plt.text(time_position + duration/2, 0.2, f"{track}", 
                        ha='center', va='center', color='black')
                
                # Move to next position
                time_position += duration
            
            # Add transition markers
            time_position = segments[0]['duration']
            for transition in transitions:
                plt.axvline(x=time_position, color='black', linestyle='--', alpha=0.5)
                plt.text(time_position, 0.9, 
                        f"{transition['type']}\n{transition['score']:.2f}", 
                        ha='center', va='center', fontsize=8)
                
                # Move to next position
                time_position += transition['duration'] + segments[transitions.index(transition)+1]['duration']
            
            # Set labels and title
            plt.title("Mashup Structure")
            plt.xlabel("Time (seconds)")
            plt.xlim(0, time_position)
            plt.ylim(0, 1)
            plt.grid(axis='x', alpha=0.3)
            
            # Remove y axis
            plt.gca().get_yaxis().set_visible(False)
            
            # Add legend
            patches = [plt.Rectangle((0,0), 1, 1, color=track_colors[track]) for track in tracks]
            plt.legend(patches, tracks, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=min(5, len(tracks)))
            
            # Save figure
            save_path.parent.mkdir(exist_ok=True, parents=True)
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
            
            logger.info(f"Mashup visualization saved to {save_path}")
        except Exception as e:
            logger.error(f"Error visualizing mashup: {str(e)}")
            logger.error(traceback.format_exc())
    
    def save_mashup_data(self, segments: List[Dict[str, Any]], transitions: List[Dict[str, Any]], 
                       save_path: Optional[Path] = None) -> None:
        """
        Save mashup data to JSON file
        
        Args:
            segments: List of segment data
            transitions: List of transition data
            save_path: Path to save the data
        """
        if not save_path:
            save_path = VISUALIZATION_DIR / "mashup_structure.json"
            
        logger.info(f"Saving mashup data to {save_path}")
        
        try:
            # Create data structure
            mashup_data = {
                'segments': segments,
                'transitions': transitions,
                'total_duration': sum(segment['duration'] for segment in segments)
            }
            
            # Save to file
            save_path.parent.mkdir(exist_ok=True, parents=True)
            with open(save_path, 'w') as f:
                json.dump(mashup_data, f, indent=2)
                
            logger.info(f"Mashup data saved to {save_path}")
        except Exception as e:
            logger.error(f"Error saving mashup data: {str(e)}")
            logger.error(traceback.format_exc())


def generate_mashup(tracks_data: Dict[str, Dict[str, Any]], transition_matrix: Dict[str, Dict[str, Dict[str, Any]]],
                  target_duration: Optional[float] = None, 
                  start_track: Optional[str] = None,
                  end_track: Optional[str] = None,
                  min_tracks: int = MIN_SONGS_IN_MASHUP,
                  max_consecutive: int = MAX_CONSECUTIVE_PHRASES_FROM_SAME_SONG,
                  randomness_boost: float = 1.0) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Top-level function to generate a mashup
    
    Args:
        tracks_data: Dictionary of track data
        transition_matrix: Matrix of possible transitions
        target_duration: Target duration in minutes
        start_track: Optional track to start with
        end_track: Optional track to end with
        min_tracks: Minimum number of tracks to include
        max_consecutive: Maximum consecutive segments from same track
        randomness_boost: Factor to increase randomness/creativity
        
    Returns:
        Tuple of (segments list, transitions list)
    """
    logger.info("Generating mashup")
    
    # Create generator
    generator = MashupGenerator(tracks_data, transition_matrix)
    
    # Convert duration to minutes if specified
    if target_duration is not None:
        target_duration = float(target_duration)
    
    # Generate the mashup
    return generator.generate_mashup(
        target_duration_minutes=target_duration,
        start_track=start_track,
        end_track=end_track,
        min_tracks=min_tracks,
        max_consecutive=max_consecutive,
        randomness_boost=randomness_boost
    )


if __name__ == "__main__":
    # Test code
    import json
    import sys
    from pathlib import Path
    from transitions_ai.src.transition_engine import TransitionEngine
    
    # Load sample data from files if available
    if len(sys.argv) > 1:
        data_dir = Path(sys.argv[1])
        tracks_data = {}
        
        # Load features and segments from JSON files
        for file_path in data_dir.glob("*.json"):
            with open(file_path, 'r') as f:
                tracks_data[file_path.stem] = json.load(f)
        
        if tracks_data:
            # Build transition matrix
            engine = TransitionEngine(tracks_data)
            transition_matrix = engine.build_transition_matrix()
            
            # Generate mashup
            segments, transitions = generate_mashup(tracks_data, transition_matrix)
            
            if segments:
                print(f"Generated mashup with {len(segments)} segments:")
                for i, segment in enumerate(segments):
                    print(f"{i+1}. {segment['track']} - {segment['type']}: {segment['duration']:.2f}s")
            else:
                print("Failed to generate mashup.")
        else:
            print("No data files found.")
    else:
        print("Please provide a directory with track data JSON files.") 