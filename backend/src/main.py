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

    segments = []

    # Load and segment all tracks
    for filename in files:
        file_path = os.path.join(processor.input_dir, filename)
        if not os.path.exists(file_path):
            continue

        # Load and normalize audio
        audio, metadata = processor.load_audio(file_path)

        # Segment the track
        track_segments = processor.segment_track(
            audio, metadata, target_length_ms=segment_length_ms
        )

        # Extract features for each segment
        for segment in track_segments:
            segment["features"] = processor.extract_features(segment)

        segments.extend(track_segments)

    if len(segments) < 2:
        return {"error": "Not enough valid segments to create transitions"}

    # Calculate transition scores
    transition_scores = processor.calculate_transition_scores(segments)

    # Generate sequence
    sequence = []
    current_segment = segments[0]  # Start with first segment
    sequence.append(current_segment)

    while len(sequence) < target_segments:
        # Get transition candidates for current segment
        candidates = transition_scores.get(current_segment["segment_id"], [])

        if not candidates:
            break

        # Get highest scoring candidate
        next_segment_id = candidates[0][0]
        next_segment = next(
            (s for s in segments if s["segment_id"] == next_segment_id), None
        )

        if not next_segment:
            break

        sequence.append(next_segment)
        current_segment = next_segment

    # Create the final mix
    final_mix = sequence[0]["audio"]

    for i in range(1, len(sequence)):
        # Create transition between segments
        transition = processor.create_transition(
            sequence[i - 1], sequence[i], crossfade_duration_ms
        )
        # Append the next segment with crossfade
        final_mix = final_mix.append(
            sequence[i]["audio"], crossfade=crossfade_duration_ms
        )

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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
