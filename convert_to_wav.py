from pydub import AudioSegment
import os
from pathlib import Path

input_dir = Path("transitions_ai/data/input")
output_dir = Path("transitions_ai/data/input")

# Create output directory if it doesn't exist
output_dir.mkdir(exist_ok=True, parents=True)

# Find all MP3 files
mp3_files = list(input_dir.glob("*.mp3"))
print(f"Found {len(mp3_files)} MP3 files")

# Convert each MP3 to WAV
for mp3_file in mp3_files:
    print(f"Converting {mp3_file.name}...")
    wav_file = output_dir / f"{mp3_file.stem}.wav"
    
    # Load MP3 and export as WAV
    audio = AudioSegment.from_mp3(mp3_file)
    audio.export(wav_file, format="wav")
    
    print(f"Created {wav_file.name}")

print("Conversion complete! You can now run the analysis with the WAV files.") 