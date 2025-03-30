import os
import logging
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"
TEMP_DIR = DATA_DIR / "temp"
LOGS_DIR = PROJECT_ROOT / "logs"
VISUALIZATION_DIR = PROJECT_ROOT / "visualizations"

# Ensure directories exist
for directory in [INPUT_DIR, OUTPUT_DIR, TEMP_DIR, LOGS_DIR, VISUALIZATION_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

# Audio analysis parameters
SAMPLE_RATE = 44100
HOP_LENGTH = 512
N_FFT = 2048
N_MELS = 128

# Phrase segmentation parameters
MIN_PHRASE_LENGTH_SECONDS = 4
MAX_PHRASE_LENGTH_SECONDS = 32
ENERGY_THRESHOLD = 0.4
ONSET_THRESHOLD = 0.4
BOUNDARY_DETECTION_SENSITIVITY = 0.6
SEGMENT_PADDING_SECONDS = 0.5

# Transition parameters
TRANSITION_TYPES = [
    'cut', 'crossfade', 'beatmatch_crossfade', 'harmonic_crossfade', 
    'filter_sweep', 'echo_out', 'spinup', 'scratch_in', 'reverse_cymbal',
    'drum_fill', 'stutter_cut', 'glitch_effect', 'tape_stop', 'beat_roll',
    'vocal_sample', 'dj_shout', 'power_down', 'bass_drop'
]
DEFAULT_TRANSITION_TYPE = 'beatmatch_crossfade'
DEFAULT_TRANSITION_DURATION_SECONDS = 4
MAX_BPM_DIFFERENCE = 20
BPM_CHANGE_TOLERANCE = 0.1
KEY_COMPATIBILITY_THRESHOLD = 0.3
ENERGY_MATCHING_WEIGHT = 0.2
HARMONIC_COMPATIBILITY_WEIGHT = 0.3
RHYTHM_COMPATIBILITY_WEIGHT = 0.2
CREATIVE_WEIGHT = 0.3

# Add unpredictability parameters
RANDOMNESS_FACTOR = 0.4
SURPRISE_TRANSITION_CHANCE = 0.35
VIBE_PATTERN_WEIGHT = 0.5

# EQ and effect parameters
TRANSITION_EQ_GAIN = {
    'low': 0.7,
    'mid': 0.8,
    'high': 0.9
}
FILTER_FREQUENCIES = {
    'lowpass': 500,
    'highpass': 2000
}
REVERB_AMOUNT = 0.3
VOLUME_NORMALIZATION_TARGET = -14  # dBFS

# Creative effect parameters
GLITCH_INTENSITY = 0.7
STUTTER_REPEATS = [2, 3, 4, 8]
TAPE_STOP_DURATION = [0.5, 1.0, 2.0]
ECHO_FEEDBACK = 0.6
FILTER_SWEEP_SPEED = [0.5, 1.0, 2.0]

# Mashup generation parameters
MIN_SONGS_IN_MASHUP = 3
MAX_CONSECUTIVE_PHRASES_FROM_SAME_SONG = 1
MIN_PHRASES_PER_SONG = 1
QUALITY_THRESHOLD = 0.6
MAX_MASHUP_LENGTH_MINUTES = 5
PATH_FINDING_HEURISTIC_WEIGHT = 1.2
MAX_BACKTRACKING_ATTEMPTS = 8

# Output format parameters
OUTPUT_FORMAT = 'mp3'
OUTPUT_BITRATE = '320k'

# Logging configuration
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(module)s - %(message)s"
LOG_FILE = LOGS_DIR / "transitions_ai.log" 