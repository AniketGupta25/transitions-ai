def _get_transition_compatibility(from_segment: Dict[str, Any], to_segment: Dict[str, Any]) -> float:
    """
    Calculate how compatible two segments are for transitioning.
    Returns a score from 0.0 (incompatible) to 1.0 (highly compatible).
    """
    from_type = from_segment.get('type', '').lower()
    to_type = to_segment.get('type', '').lower()
    
    # Define preferred segment transitions
    transition_scores = {
        # Natural song progression
        ('intro', 'verse'): 0.9,
        ('verse', 'chorus'): 0.9,
        ('chorus', 'verse'): 0.8,
        ('verse', 'bridge'): 0.8,
        ('bridge', 'chorus'): 0.9,
        ('chorus', 'outro'): 0.8,
        
        # Less natural but acceptable
        ('verse', 'verse'): 0.7,
        ('chorus', 'chorus'): 0.7,
        ('bridge', 'verse'): 0.6,
        ('bridge', 'bridge'): 0.5,
        
        # Avoid these transitions
        ('outro', 'intro'): 0.2,
        ('chorus', 'intro'): 0.2,
        ('bridge', 'intro'): 0.2,
        ('outro', 'chorus'): 0.3,
        ('intro', 'outro'): 0.1,
        ('intro', 'chorus'): 0.3,
    }
    
    # Get base compatibility score
    base_score = transition_scores.get((from_type, to_type), 0.4)  # Default to 0.4 for unknown combinations
    
    # Adjust score based on musical qualities
    from_energy = from_segment.get('energy', 0.5)
    to_energy = to_segment.get('energy', 0.5)
    energy_diff = abs(from_energy - to_energy)
    
    # Penalize large energy jumps
    if energy_diff > 0.3:
        base_score *= (1 - (energy_diff - 0.3))
    
    # Consider BPM compatibility
    from_bpm = from_segment.get('bpm', 120)
    to_bpm = to_segment.get('bpm', 120)
    bpm_ratio = min(from_bpm, to_bpm) / max(from_bpm, to_bpm)
    
    # Penalize BPM mismatches unless they're related (e.g. double/half time)
    if bpm_ratio < 0.9 and bpm_ratio not in (0.5, 0.25, 0.75):
        base_score *= bpm_ratio
    
    return max(0.0, min(1.0, base_score))

def _select_next_segment(current_segment: Dict[str, Any], 
                        available_segments: List[Dict[str, Any]], 
                        energy_target: float,
                        used_tracks: Set[str],
                        min_segments_before_repeat: int = 3) -> Dict[str, Any]:
    """
    Select the next segment based on musical compatibility and energy flow
    """
    best_segment = None
    best_score = -1
    
    for segment in available_segments:
        # Skip segments from recently used tracks to avoid quick repeats
        if segment['track'] in used_tracks:
            continue
            
        # Calculate base compatibility score
        compatibility = _get_transition_compatibility(current_segment, segment)
        
        # Calculate energy score (how well it matches target energy)
        energy_score = 1 - abs(segment.get('energy', 0.5) - energy_target)
        
        # Calculate variety score (prefer different tracks)
        variety_score = 0.8 if segment['track'] != current_segment['track'] else 0.3
        
        # Combine scores with weights
        total_score = (
            compatibility * 0.5 +  # Compatibility is most important
            energy_score * 0.3 +  # Energy flow is next
            variety_score * 0.2    # Variety adds spice
        )
        
        # Update best if this is better
        if total_score > best_score:
            best_score = total_score
            best_segment = segment
    
    # If no good matches found, relax restrictions
    if best_score < 0.4:
        # Clear used tracks restriction and try again with lower standards
        return _select_next_segment(current_segment, available_segments, energy_target, set(), 1)
    
    return best_segment

def _get_musical_similarity(from_segment: Dict[str, Any], to_segment: Dict[str, Any]) -> float:
    """
    Calculate musical similarity between segments based on key, BPM, and energy.
    Returns a score from 0.0 (completely different) to 1.0 (very similar).
    """
    # Get musical properties
    from_key = from_segment.get('key', 'C major')
    to_key = to_segment.get('key', 'C major')
    from_bpm = from_segment.get('bpm', 120)
    to_bpm = to_segment.get('bpm', 120)
    from_energy = from_segment.get('energy', 0.5)
    to_energy = to_segment.get('energy', 0.5)
    
    # Calculate BPM compatibility (allow for double/half time)
    bpm_ratio = min(from_bpm, to_bpm) / max(from_bpm, to_bpm)
    if bpm_ratio < 0.9 and bpm_ratio not in (0.5, 0.25, 0.75):
        return 0.0  # BPMs are too different
    
    # Calculate key compatibility
    key_compatibility = 1.0 if from_key == to_key else 0.5  # For now simple match/no match
    
    # Calculate energy similarity
    energy_diff = abs(from_energy - to_energy)
    energy_compatibility = max(0.0, 1.0 - energy_diff)
    
    # Combine scores with weights
    similarity = (
        key_compatibility * 0.4 +    # Key matching is important
        energy_compatibility * 0.6    # Energy flow is crucial
    )
    
    return similarity

def _find_similar_segments(current_segment: Dict[str, Any], 
                          available_segments: List[Dict[str, Any]], 
                          min_similarity: float = 0.7) -> List[Dict[str, Any]]:
    """Find segments that are musically similar to the current segment"""
    similar_segments = []
    
    for segment in available_segments:
        if segment['track'] == current_segment['track']:
            continue  # Skip segments from same track
            
        similarity = _get_musical_similarity(current_segment, segment)
        if similarity >= min_similarity:
            similar_segments.append((segment, similarity))
    
    # Sort by similarity
    similar_segments.sort(key=lambda x: x[1], reverse=True)
    return [s[0] for s in similar_segments]

def create_mashup(tracks_data: Dict[str, Dict[str, Any]], 
                 energy_flow: str = 'wave',
                 preferred_transition_types: List[str] = None,
                 target_duration: float = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Create a mashup with improved musical transitions
    """
    if not tracks_data:
        raise ValueError("No tracks provided")
    
    # Initialize segments pool
    all_segments = []
    for track_name, track_data in tracks_data.items():
        segments = track_data.get('segments', [])
        for segment in segments:
            segment['track'] = track_name
            segment['bpm'] = track_data.get('bpm', 120)
            segment['key'] = track_data.get('key', 'C major')
            all_segments.append(segment)
    
    if not all_segments:
        raise ValueError("No segments found in tracks")
    
    # Initialize mashup structure
    mashup_segments = []
    transitions = []
    used_tracks = set()
    current_duration = 0
    
    # Find a good starting segment (prefer intros with moderate energy)
    start_segments = [s for s in all_segments 
                     if s['type'].lower() == 'intro' 
                     and 0.4 <= s.get('energy', 0.5) <= 0.6]
    if not start_segments:
        start_segments = [s for s in all_segments if s['type'].lower() == 'intro']
    if not start_segments:
        start_segments = all_segments
    
    current_segment = start_segments[0]  # Just take first suitable intro
    mashup_segments.append(current_segment)
    used_tracks.add(current_segment['track'])
    current_duration += current_segment['duration']
    
    # Track the current musical sequence
    current_sequence = ['intro']
    desired_sequences = [
        ['intro', 'verse', 'chorus', 'verse', 'chorus', 'outro'],
        ['intro', 'verse', 'chorus', 'bridge', 'chorus', 'outro'],
        ['intro', 'verse', 'chorus', 'verse', 'bridge', 'chorus', 'outro']
    ]
    
    # Create the mashup
    while (not target_duration or current_duration < target_duration) and len(mashup_segments) < 12:
        # Determine next desired segment type
        next_type = None
        for sequence in desired_sequences:
            if sequence[:len(current_sequence)] == current_sequence:
                if len(sequence) > len(current_sequence):
                    next_type = sequence[len(current_sequence)]
                    break
        
        if not next_type:
            # If we can't follow a sequence, default to verse-chorus alternation
            last_type = current_sequence[-1]
            if last_type == 'verse':
                next_type = 'chorus'
            elif last_type == 'chorus':
                next_type = 'verse'
            else:
                next_type = 'verse'
        
        # Find similar segments of the desired type
        candidates = [s for s in all_segments if s['type'].lower() == next_type]
        similar_segments = _find_similar_segments(current_segment, candidates)
        
        if not similar_segments:
            # If no similar segments found, try with lower similarity threshold
            similar_segments = _find_similar_segments(current_segment, candidates, min_similarity=0.5)
        
        if not similar_segments:
            # If still no segments found, just take any segment of desired type
            similar_segments = candidates
        
        if not similar_segments:
            break  # No suitable segments found
        
        # Select next segment (prefer unused tracks)
        next_segment = None
        for segment in similar_segments:
            if segment['track'] not in used_tracks:
                next_segment = segment
                break
        if not next_segment:
            next_segment = similar_segments[0]
        
        # Create appropriate transition
        transition_type = _select_transition_type(
            current_segment, 
            next_segment,
            preferred_transition_types
        )
        
        # Add transition
        transition = {
            'type': transition_type,
            'duration': _get_transition_duration(current_segment, next_segment),
            'from_segment': f"{current_segment['track']}_{current_segment['type']}",
            'to_segment': f"{next_segment['track']}_{next_segment['type']}",
            'from_energy': current_segment.get('energy', 0.5),
            'to_energy': next_segment.get('energy', 0.5),
            'bpm': (current_segment.get('bpm', 120) + next_segment.get('bpm', 120)) / 2
        }
        transitions.append(transition)
        
        # Update state
        mashup_segments.append(next_segment)
        used_tracks.add(next_segment['track'])
        if len(used_tracks) >= len(tracks_data):
            used_tracks = {next_segment['track']}  # Reset but keep current
        
        current_segment = next_segment
        current_duration += next_segment['duration']
        current_sequence.append(next_type)
    
    return mashup_segments, transitions

def _select_transition_type(from_segment: Dict[str, Any], 
                          to_segment: Dict[str, Any],
                          preferred_types: List[str] = None) -> str:
    """
    Select appropriate transition type based on musical context
    """
    if not preferred_types:
        preferred_types = ['beatmatch_crossfade', 'filter_sweep', 'harmonic_crossfade']
    
    from_type = from_segment.get('type', '').lower()
    to_type = to_segment.get('type', '').lower()
    
    # Use musical context to select transition type
    if from_type == to_type:
        # Same section type - use harmonic crossfade for smooth blend
        if 'harmonic_crossfade' in preferred_types:
            return 'harmonic_crossfade'
    
    if from_type in ('chorus', 'drop') and to_type in ('verse', 'bridge'):
        # Transitioning from high to low energy sections
        if 'filter_sweep' in preferred_types:
            return 'filter_sweep'
    
    if from_type in ('verse', 'bridge') and to_type in ('chorus', 'drop'):
        # Building energy
        if 'harmonic_crossfade' in preferred_types:
            return 'harmonic_crossfade'
    
    # Default to beatmatched crossfade
    return preferred_types[0] if preferred_types else 'beatmatch_crossfade'

def _get_transition_duration(from_segment: Dict[str, Any], 
                           to_segment: Dict[str, Any]) -> float:
    """
    Calculate appropriate transition duration based on musical context
    """
    # Get BPMs
    from_bpm = from_segment.get('bpm', 120)
    to_bpm = to_segment.get('bpm', 120)
    avg_bpm = (from_bpm + to_bpm) / 2
    
    # Calculate beats for transition based on musical context
    from_type = from_segment.get('type', '').lower()
    to_type = to_segment.get('type', '').lower()
    
    if from_type == to_type:
        beats = 8  # Longer blend for similar sections
    elif from_type in ('chorus', 'drop') and to_type in ('verse', 'bridge'):
        beats = 8  # Longer transition when going from high to low energy
    elif from_type in ('verse', 'bridge') and to_type in ('chorus', 'drop'):
        beats = 4  # Shorter build into high energy sections
    else:
        beats = 6  # Default transition length
    
    # Convert beats to seconds
    duration = (beats * 60) / avg_bpm
    
    # Ensure reasonable bounds
    return min(max(duration, 2.0), 8.0)
