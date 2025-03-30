import logging
import os
import sys
import time
from pathlib import Path
import colorlog
from typing import Optional

from transitions_ai.src.config import LOG_LEVEL, LOG_FORMAT, LOG_FILE, LOGS_DIR

class TransitionsLogger:
    """
    Custom logger for the transitions_ai project that provides comprehensive
    logging with colored output to console and detailed logs to file.
    """
    
    def __init__(self, name: str, log_file: Optional[Path] = None):
        """
        Initialize the logger with the given name and optional log file path.
        
        Args:
            name: Name of the logger
            log_file: Path to the log file. If None, uses the default from config.
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(LOG_LEVEL)
        self.logger.handlers = []  # Clear any existing handlers
        
        # Set up colored console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(LOG_LEVEL)
        
        color_formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s - %(levelname)s - %(module)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )
        console_handler.setFormatter(color_formatter)
        self.logger.addHandler(console_handler)
        
        # Set up file handler
        if log_file is None:
            log_file = LOG_FILE
            
        # Create timestamp-based log file if not specified
        if log_file == LOG_FILE:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            log_file = LOGS_DIR / f"transitions_ai_{timestamp}.log"
            
        log_file.parent.mkdir(exist_ok=True, parents=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(LOG_LEVEL)
        
        file_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(module)s - %(funcName)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        self.logger.info(f"Logger initialized: {name}")
        
    def get_logger(self):
        """Return the configured logger instance."""
        return self.logger


# Create a default logger instance
default_logger = TransitionsLogger("transitions_ai").get_logger()

# Helper functions to easily access the default logger
def get_logger(name: str = "transitions_ai"):
    """Get a logger with the given name."""
    return TransitionsLogger(name).get_logger()

def log_audio_analysis(track_name: str, features: dict):
    """Log audio analysis results."""
    logger = get_logger("audio_analysis")
    logger.info(f"Analysis completed for {track_name}")
    for feature, value in features.items():
        if isinstance(value, (int, float, str)):
            logger.debug(f"{feature}: {value}")
        else:
            logger.debug(f"{feature}: [complex data]")
            
def log_segmentation(track_name: str, segments: list):
    """Log segmentation results."""
    logger = get_logger("segmentation")
    logger.info(f"Segmentation completed for {track_name}")
    logger.info(f"Found {len(segments)} segments")
    for i, segment in enumerate(segments):
        logger.debug(f"Segment {i+1}: {segment.get('type', 'unknown')} - "
                    f"Start: {segment.get('start', 0):.2f}s, "
                    f"End: {segment.get('end', 0):.2f}s, "
                    f"Duration: {segment.get('duration', 0):.2f}s")
        
def log_transition(from_track: str, to_track: str, transition_type: str, score: float):
    """Log transition information."""
    logger = get_logger("transition")
    logger.info(f"Transition: {from_track} → {to_track}")
    logger.info(f"Type: {transition_type}, Score: {score:.2f}")
    
def log_mashup_generation(structure: list, total_duration: float, num_transitions: int):
    """Log mashup generation information."""
    logger = get_logger("mashup")
    logger.info(f"Mashup generated: {total_duration:.2f}s with {num_transitions} transitions")
    for i, segment in enumerate(structure):
        logger.debug(f"Segment {i+1}: {segment.get('track', 'unknown')} - "
                    f"{segment.get('phrase_type', 'unknown')}") 