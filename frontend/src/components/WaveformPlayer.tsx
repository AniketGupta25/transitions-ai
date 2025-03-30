import React, { useEffect, useRef, useState } from 'react';

const WaveformPlayer = ({ 
  audioUrl = '/song.wav', 
  backgroundColor = 'black', 
  peakColor = '#FFA500',
  progressPeakColor = '#FFD700',
  containerWidth = '100%',
  containerHeight = '80px'
}) => {
  const canvasRef = useRef(null);
  const audioRef = useRef(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const audioContextRef = useRef(null);
  const [audioBuffer, setAudioBuffer] = useState(null);
  const [volume, setVolume] = useState(1);
  const animationRef = useRef(null);
  const [isLoading, setIsLoading] = useState(false);

  // Initialize audio context
  useEffect(() => {
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    audioContextRef.current = new AudioContext();

    return () => {
      // Cleanup function
      if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
        audioContextRef.current.close();
      }
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, []);

  // Load audio when URL changes
  useEffect(() => {
    if (!audioUrl || !audioContextRef.current) return;

    const loadAudio = async () => {
      setIsLoading(true);
      setIsPlaying(false);
      setCurrentTime(0);
      setDuration(0);
      setAudioBuffer(null);

      try {
        if (audioRef.current) {
          audioRef.current.pause();
          audioRef.current.currentTime = 0;
        }

        const response = await fetch(audioUrl);
        const arrayBuffer = await response.arrayBuffer();
        const buffer = await audioContextRef.current.decodeAudioData(arrayBuffer);
        
        setAudioBuffer(buffer);
        setDuration(buffer.duration);
        drawWaveform(buffer, 0);
      } catch (error) {
        console.error('Error loading audio:', error);
      } finally {
        setIsLoading(false);
      }
    };

    loadAudio();
  }, [audioUrl]);

  // Draw waveform on canvas
  const drawWaveform = (buffer, progressPosition = 0) => {
    if (!canvasRef.current) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const data = buffer.getChannelData(0);
    const width = canvas.width;
    const height = canvas.height;
    const step = Math.ceil(data.length / width);
    const amp = height / 2;
    
    ctx.fillStyle = backgroundColor;
    ctx.fillRect(0, 0, width, height);
    
    for (let i = 0; i < width; i++) {
      let min = 1.0;
      let max = -1.0;
      
      for (let j = 0; j < step; j++) {
        const index = (i * step) + j;
        if (index < data.length) {
          const datum = data[index];
          if (datum < min) min = datum;
          if (datum > max) max = datum;
        }
      }
      
      const peakHeight = Math.max(Math.abs(min), Math.abs(max)) * amp;
      
      ctx.fillStyle = i <= progressPosition ? progressPeakColor : peakColor;
      ctx.fillRect(i, amp - peakHeight, 1, peakHeight);
    }
  };

  // Update waveform progress
  const updateWaveformProgress = () => {
    if (!canvasRef.current || !audioBuffer || !duration) return;
    
    const progress = currentTime / duration;
    const progressPos = Math.floor(canvasRef.current.width * progress);
    drawWaveform(audioBuffer, progressPos);
  };

  // Play/pause toggle with AudioContext resume
  const togglePlay = async () => {
    if (!audioRef.current || !audioContextRef.current) return;
    
    try {
      // Resume AudioContext if suspended
      if (audioContextRef.current.state === 'suspended') {
        await audioContextRef.current.resume();
      }

      if (isPlaying) {
        audioRef.current.pause();
        setIsPlaying(false);
      } else {
        await audioRef.current.play();
        setIsPlaying(true);
        startAnimation();
      }
    } catch (error) {
      console.error('Playback error:', error);
    }
  };

  // Animation loop for smooth progress updates
  const startAnimation = () => {
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
    }
    
    const animate = () => {
      if (audioRef.current) {
        setCurrentTime(audioRef.current.currentTime);
      }
      animationRef.current = requestAnimationFrame(animate);
    };
    
    animationRef.current = requestAnimationFrame(animate);
  };

  // Handle audio element events
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    const handleTimeUpdate = () => setCurrentTime(audio.currentTime);
    const handleAudioEnded = () => {
      setIsPlaying(false);
      setCurrentTime(0);
      cancelAnimationFrame(animationRef.current);
    };

    audio.addEventListener('timeupdate', handleTimeUpdate);
    audio.addEventListener('ended', handleAudioEnded);

    return () => {
      audio.removeEventListener('timeupdate', handleTimeUpdate);
      audio.removeEventListener('ended', handleAudioEnded);
      cancelAnimationFrame(animationRef.current);
    };
  }, []);

  // Update waveform when currentTime changes
  useEffect(() => {
    updateWaveformProgress();
  }, [currentTime, audioBuffer]);

  // Volume control
  const handleVolumeChange = (e) => {
    const value = parseFloat(e.target.value);
    setVolume(value);
    if (audioRef.current) {
      audioRef.current.volume = value;
    }
  };

  // Format time as MM:SS
  const formatTime = (time) => {
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    return `${minutes}:${seconds < 10 ? '0' : ''}${seconds}`;
  };

  // Handle clicking on waveform to seek
  const handleCanvasClick = (e) => {
    if (!audioRef.current || !canvasRef.current || !audioBuffer || !duration) return;
    
    const rect = canvasRef.current.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickPosition = clickX / rect.width;
    const newTime = duration * clickPosition;
    
    audioRef.current.currentTime = newTime;
    setCurrentTime(newTime);
    drawWaveform(audioBuffer, Math.floor(canvasRef.current.width * clickPosition));
  };

  return (
    <div className="waveform-player" style={{ 
      background: 'transparent', 
      width: containerWidth,
      maxWidth: '100%'
    }}>
      <div style={{ 
        width: '100%', 
        height: containerHeight,
        background: 'black',
        position: 'relative',
      }}>
        <canvas 
          ref={canvasRef} 
          width={1000} 
          height={parseInt(containerHeight)}
          onClick={handleCanvasClick}
          style={{ 
            cursor: 'pointer', 
            width: '100%',
            height: '100%',
            background: 'black',
          }}
        />
        {isLoading && (
          <div style={{
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: 'white',
            backgroundColor: 'rgba(0,0,0,0.7)'
          }}>
            Loading...
          </div>
        )}
      </div>
      
      <div className="controls" style={{ 
        display: 'flex', 
        alignItems: 'center', 
        padding: '10px 0',
        background: 'black',
        color: 'white'
      }}>
        <button 
          onClick={togglePlay}
          disabled={isLoading || !audioBuffer}
          style={{
            padding: '8px 16px',
            marginRight: '10px',
            backgroundColor: '#4CAF50',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            opacity: isLoading || !audioBuffer ? 0.5 : 1
          }}
        >
          {isPlaying ? 'Pause' : 'Play'}
        </button>
        
        <div className="time" style={{ marginRight: '15px', color: 'white' }}>
          {formatTime(currentTime)} / {formatTime(duration)}
        </div>
        
        <div className="volume-control" style={{ display: 'flex', alignItems: 'center' }}>
          <span style={{ marginRight: '5px', color: 'white' }}>Volume:</span>
          <input
            type="range"
            min="0"
            max="1"
            step="0.01"
            value={volume}
            onChange={handleVolumeChange}
            disabled={isLoading || !audioBuffer}
            style={{ 
              width: '100px',
              accentColor: '#4CAF50'
            }}
          />
        </div>
      </div>
      
      <audio ref={audioRef} src={audioUrl} preload="metadata" />
    </div>
  );
};

export default WaveformPlayer;