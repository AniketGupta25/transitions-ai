# Audio Transition Platform

An intelligent system for creating smooth transitions between audio tracks. This platform analyzes audio features like tempo, key, energy, and timbre to create seamless transitions between compatible segments of music.

## Features

- Audio file upload and management
- Beat-aligned track segmentation
- Intelligent transition scoring based on multiple audio features:
  - Tempo matching
  - Harmonic compatibility (using chroma features)
  - Timbre similarity (using MFCCs)
  - Energy level matching
- Automatic sequence generation with optimal transitions
- Smooth crossfaded transitions between segments

## Setup

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Create necessary directories:
```bash
mkdir -p input output temp
```

3. Start the server:
```bash
python src/main.py
```

The server will start on `http://localhost:8000`.

## API Endpoints

### `POST /upload`
Upload an audio file.
- Request: `multipart/form-data` with file
- Response: `{"filename": string, "status": string}`

### `POST /create-mix`
Create a mix from uploaded audio files.
- Request body:
```json
{
  "files": ["file1.mp3", "file2.wav"],
  "target_segments": 15,
  "segment_length_ms": 90000,
  "crossfade_duration_ms": 3000
}
```
- Response:
```json
{
  "status": "success",
  "output_file": "mix_filename.wav",
  "segments_used": 15,
  "duration_ms": 1350000
}
```

### `GET /list-files`
List all available audio files.
- Response: Array of file objects with filename and size

## Usage Example

1. Upload audio files:
```bash
curl -X POST -F "file=@song1.mp3" http://localhost:8000/upload
curl -X POST -F "file=@song2.mp3" http://localhost:8000/upload
```

2. Create a mix:
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"files": ["song1.mp3", "song2.mp3"]}' \
  http://localhost:8000/create-mix
```

3. The mixed audio file will be available in the `output` directory.

## Technical Details

### Audio Analysis
- Beat detection using librosa
- Chroma feature extraction for harmonic analysis
- MFCC analysis for timbre matching
- RMS energy analysis for volume matching
- Normalized audio features for consistent scoring

### Transition Scoring
Transitions are scored based on weighted criteria:
- Tempo matching (40%)
- Harmonic compatibility (30%)
- Timbre similarity (20%)
- Energy level matching (10%)

### Performance Considerations
- Temporary files are cleaned up automatically
- Audio processing is done in segments to manage memory
- Feature extraction results are cached per session

## Requirements

- Python 3.8+
- FFmpeg (for audio file handling)
- Sufficient disk space for audio processing
- Memory requirements depend on audio file sizes and number of segments

## License

MIT License 