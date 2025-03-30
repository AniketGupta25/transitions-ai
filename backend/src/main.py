import os
from typing import List, Dict, Optional
from datetime import datetime
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydub import AudioSegment
from audio_processor import AudioProcessor

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize AudioProcessor
processor = AudioProcessor(
    input_dir="./input", output_dir="./output", temp_dir="./temp"
)


@app.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    """Upload an audio file."""
    # Save the uploaded file
    file_path = os.path.join(processor.input_dir, file.filename)
    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)

    return {"filename": file.filename, "status": "uploaded"}


@app.post("/create-mix")
async def create_mix(
    target_segments: int = 15,
    segment_length_ms: int = 90000,
    crossfade_duration_ms: int = 3000,
    similarity_threshold: float = 85.0,
    min_segments_per_track: int = 1,
):
    """Create a mix from the specified audio files or all files if none specified."""
    # If no files specified, use all audio files in input directory
    files = [
        f
        for f in os.listdir(processor.input_dir)
        if f.lower().endswith((".mp3", ".wav", ".m4a", ".aac"))
    ]

    if len(files) == 0:
        return {"error": "No audio files found in input directory"}

    print("\n=== Starting Mix Creation ===")
    print(f"Found {len(files)} input files: {', '.join(files)}")

    segments = []

    # Load and segment all tracks
    for filename in files:
        file_path = os.path.join(processor.input_dir, filename)
        if not os.path.exists(file_path):
            continue

        print(f"\nProcessing track: {filename}")
        audio, metadata = processor.load_audio(file_path)
        track_segments = processor.segment_track(
            audio, metadata, target_length_ms=segment_length_ms
        )
        print(f"  ├── Created {len(track_segments)} segments")

        # Extract features for each segment
        for i, segment in enumerate(track_segments, 1):
            segment["features"] = processor.extract_features(segment)
        print(f"  └── Extracted features for all segments")

        segments.extend(track_segments)

    if len(segments) < 2:
        return {"error": "Not enough valid segments to create transitions"}

    # Calculate transition and similarity scores
    transition_scores, similarity_scores = processor.calculate_transition_scores(
        segments
    )

    def format_ms_to_timestamp(ms):
        """Convert milliseconds to MM:SS format"""
        total_seconds = ms // 1000
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        return f"{minutes:02d}:{seconds:02d}"

    # Phase 1: Generate Sequence
    print("\n=== Phase 1: Generating Sequence ===")
    sequence = []
    current_segment = segments[0]
    used_segments = set()
    invalid_segments = set()
    segments_per_track = {}
    current_position_ms = 0

    print(
        f"Starting with: {current_segment['original_track']} at {format_ms_to_timestamp(current_position_ms)}"
    )

    # Add first segment and invalidate similar ones
    sequence.append(current_segment)
    used_segments.add(current_segment["segment_id"])
    segments_per_track[current_segment["original_track"]] = 1

    # Initial invalidation
    similar_segments = [
        (s_id, score)
        for s_id, score in similarity_scores[current_segment["segment_id"]]
        if score >= similarity_threshold
    ]
    if similar_segments:
        print(f"Invalidating {len(similar_segments)} similar segments:")
        for s_id, score in similar_segments:
            segment = next((s for s in segments if s["segment_id"] == s_id), None)
            if segment:
                print(
                    f"  └── From {segment['original_track']} (similarity: {score:.1f}%)"
                )
                invalid_segments.add(s_id)

    sequence_step = 1
    while len(sequence) < target_segments:
        print(f"\n  Step {sequence_step}:")
        candidates = transition_scores.get(current_segment["segment_id"], [])
        valid_candidates = []

        for candidate_id, score in candidates:
            candidate = next(
                (s for s in segments if s["segment_id"] == candidate_id), None
            )
            if not candidate:
                continue

            # Check if candidate is valid
            if (
                candidate_id not in used_segments
                and candidate_id not in invalid_segments
                and (
                    segments_per_track.get(candidate["original_track"], 0)
                    < min_segments_per_track
                    or all(
                        count >= min_segments_per_track
                        for count in segments_per_track.values()
                    )
                )
            ):
                valid_candidates.append((candidate, score))

        print(f"    ├── Found {len(valid_candidates)} valid candidates")

        if not valid_candidates:
            # If no valid candidates, try to find a segment from underrepresented track
            underrepresented_tracks = [
                track
                for track, count in segments_per_track.items()
                if count < min_segments_per_track
            ]

            if underrepresented_tracks:
                print(
                    f"    ├── Looking for segments from underrepresented tracks: {', '.join(underrepresented_tracks)}"
                )
                available_segments = [
                    s
                    for s in segments
                    if s["original_track"] in underrepresented_tracks
                    and s["segment_id"] not in used_segments
                    and s["segment_id"] not in invalid_segments
                ]

                if available_segments:
                    next_segment = available_segments[0]
                    print(
                        f"    └── Selected segment from underrepresented track: {next_segment['original_track']}"
                    )
                    sequence.append(next_segment)
                    used_segments.add(next_segment["segment_id"])
                    segments_per_track[next_segment["original_track"]] = (
                        segments_per_track.get(next_segment["original_track"], 0) + 1
                    )
                    current_segment = next_segment
                    sequence_step += 1
                    continue
            print("    └── No valid candidates found, ending sequence")
            break

        next_segment, score = max(valid_candidates, key=lambda x: x[1])
        print(
            f"    └── Selected segment from {next_segment['original_track']} (score: {score:.2f})"
        )

        sequence.append(next_segment)
        used_segments.add(next_segment["segment_id"])
        segments_per_track[next_segment["original_track"]] = (
            segments_per_track.get(next_segment["original_track"], 0) + 1
        )

        # Invalidate similar segments
        similar_segments = [
            (s_id, score)
            for s_id, score in similarity_scores[next_segment["segment_id"]]
            if score >= similarity_threshold
        ]
        if similar_segments:
            print(f"Invalidating {len(similar_segments)} similar segments:")
            for s_id, score in similar_segments:
                segment = next((s for s in segments if s["segment_id"] == s_id), None)
                if segment:
                    print(
                        f"  └── From {segment['original_track']} (similarity: {score:.1f}%)"
                    )
                    invalid_segments.add(s_id)

        current_segment = next_segment
        sequence_step += 1

    print("\nFinal sequence stats:")
    for track, count in segments_per_track.items():
        print(f"  ├── {track}: {count} segments")
    print(f"  └── Total segments in mix: {len(sequence)}")

    # Phase 2: Create Mix
    print("\n=== Phase 2: Creating Mix ===")
    final_mix = sequence[0]["audio"]
    current_position_ms = len(sequence[0]["audio"])

    for i in range(1, len(sequence)):
        next_seg = sequence[i]
        print(
            f"Mixing segment {i+1}/{len(sequence)} at {format_ms_to_timestamp(current_position_ms)}"
        )
        final_mix = final_mix.append(next_seg["audio"], crossfade=crossfade_duration_ms)
        current_position_ms += len(next_seg["audio"]) - crossfade_duration_ms

    # Export the mix
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"mix_{len(sequence)}segments_{timestamp}.wav"
    output_path = os.path.join(processor.output_dir, output_filename)
    final_mix.export(output_path, format="wav")

    return {
        "status": "success",
        "output_file": output_filename,
        "segments_used": len(sequence),
        "duration_ms": len(final_mix),
        "input_files": files,
    }


@app.get("/list-files")
async def list_files():
    """List all available audio files."""
    files = []
    for filename in os.listdir(processor.input_dir):
        if filename.lower().endswith((".mp3", ".wav", ".m4a", ".aac")):
            file_path = os.path.join(processor.input_dir, filename)
            files.append(
                {"filename": filename, "size_bytes": os.path.getsize(file_path)}
            )
    return files


class SegmentTracker:
    def __init__(self, similarity_threshold: float):
        self.used_segments = set()
        self.ineligible_segments = (
            set()
        )  # combines used, similar, and temporally invalid segments
        self.segments_per_track = {}
        self.latest_position_per_track = (
            {}
        )  # Track the latest temporal position used in each track
        self.similarity_threshold = similarity_threshold

    def mark_segment_used(
        self, segment: Dict, similarity_scores: Dict, segments: List[Dict]
    ):
        """Mark a segment as used and invalidate similar segments and earlier segments from same track"""
        segment_id = segment["segment_id"]
        track = segment["original_track"]

        # Mark the segment itself
        self.used_segments.add(segment_id)
        self.ineligible_segments.add(segment_id)

        # Update track count
        self.segments_per_track[track] = self.segments_per_track.get(track, 0) + 1

        # Update latest position for this track
        self.latest_position_per_track[track] = segment["start_time_ms"]

        # Mark similar segments as ineligible
        similar_segments = [
            (s_id, score)
            for s_id, score in similarity_scores[segment_id]
            if score >= self.similarity_threshold
        ]

        print(f"\nMarking ineligible segments for {track}:")
        print(f"  ├── Segment {segment_id} marked as used")

        # Mark all earlier segments from same track as ineligible
        earlier_segments = [
            s
            for s in segments
            if s["original_track"] == track
            and s["start_time_ms"] < segment["start_time_ms"]
        ]
        if earlier_segments:
            print(
                f"  ├── Marking {len(earlier_segments)} earlier segments as ineligible"
            )
            for s in earlier_segments:
                self.ineligible_segments.add(s["segment_id"])
                print(
                    f"  │   └── Segment at {format_ms_to_timestamp(s['start_time_ms'])}"
                )

        if similar_segments:
            print(f"  ├── Found {len(similar_segments)} similar segments:")
            for s_id, score in similar_segments:
                self.ineligible_segments.add(s_id)
                print(f"  │   └── {s_id} (similarity: {score:.1f}%)")

        print(f"  └── Total ineligible segments: {len(self.ineligible_segments)}")

    def is_eligible(self, segment: Dict) -> bool:
        """Check if a segment is eligible for use"""
        # Check if segment is already ineligible
        if segment["segment_id"] in self.ineligible_segments:
            return False

        # Check temporal constraint - segment must be after latest used position for its track
        track = segment["original_track"]
        if track in self.latest_position_per_track:
            if segment["start_time_ms"] < self.latest_position_per_track[track]:
                return False

        return True


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
