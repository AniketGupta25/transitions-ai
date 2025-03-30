# Transitions AI - Automated DJ Mashup Generator

Transitions AI is a Python project that automatically generates high-quality DJ mashups from multiple input songs by analyzing and intelligently combining musical phrases.


### IMMEDIATE RUN INSTRUCTIONS

```
cd frontend
npm install
npm rn dev
```

Backend lib is /transitions-ai. Frontend code held in /frontend.
The main.py is entrypoint for the backend. Currently run locally.

## Features

- **Audio Analysis**: Extract key information from tracks (BPM, key, energy levels, etc.)
- **Phrase Segmentation**: Automatically identify distinct musical sections (intro, verse, chorus, bridge, etc.)
- **Transition Engine**: Create seamless transitions between compatible phrases
- **Mashup Generation**: Create cohesive DJ-quality mashups using graph-based path finding

## Project Structure

```
transitions_ai/
├── data/
│   ├── input/      # Place input audio files here
│   ├── output/     # Generated mashups appear here
│   └── temp/       # Temporary files and caches
├── logs/           # Detailed logging of the process
├── src/            # Source code
│   ├── audio_analysis.py        # Audio feature extraction
│   ├── phrase_segmentation.py   # Musical phrase detection
│   ├── transition_engine.py     # Creates transitions between tracks
│   ├── mashup_generator.py      # Creates the final mashup
│   ├── audio_renderer.py        # Renders audio output
│   ├── config.py                # Configuration parameters
│   └── logger.py                # Logging utilities
├── visualizations/  # Visual representations of analysis
└── tests/           # Unit tests
```

## Installation

1. Create a Python virtual environment:
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install Cython first (required for some audio dependencies):
```
pip install Cython
```

3. Install requirements:
```
pip install -r requirements.txt
```

## Usage

1. Place your audio files (WAV format recommended) in the `transitions_ai/data/input` directory.

2. Run the main script:
```
python main.py
```

3. The script will:
   - Analyze all audio files in the input directory
   - Segment each track into musical phrases
   - Identify compatible transitions between phrases
   - Generate optimized mashups
   - Save the results in the output directory

## Customization

You can customize the behavior by modifying parameters in `transitions_ai/src/config.py`:

- Adjust min/max phrase lengths
- Configure transition preferences 
- Set BPM change tolerances
- Specify key compatibility requirements
- And many more options

## Visualizations

The system generates visualizations to help understand:
- Audio features (waveform, beat tracking, energy levels)
- Phrase segmentation (identified sections like verse, chorus, etc.)
- Transition compatibility between phrases
- Overall mashup structure

## Requirements

- Python 3.8+
- librosa
- pydub
- numpy
- matplotlib
- scipy
- networkx
- and other dependencies in requirements.txt

## License

MIT 
