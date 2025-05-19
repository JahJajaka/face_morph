#!/usr/bin/env python
"""
Beat and rhythmic feature extraction tool for face morphing synchronization.
Analyzes audio files and extracts beat timestamps and rhythmic features to CSV files.

Usage:
  python beat_extraction.py --input song.mp3 --output beats.csv [--threshold 0.1] [--min_bpm 60] [--max_bpm 180]
  python beat_extraction.py --input sigma_boy.mp3 --output beats.csv --extract_rhythmic
"""

import argparse
import csv
import librosa
import numpy as np
import os
import soundfile as sf

def extract_beats(audio_file, threshold=0.4, min_bpm=60, max_bpm=180, duration=None):
    """
    Extract beat timestamps from an audio file.
    
    Args:
        audio_file (str): Path to the audio file
        threshold (float): Onset strength threshold (higher means fewer beats)
        min_bpm (int): Minimum tempo in beats per minute
        max_bpm (int): Maximum tempo in beats per minute
        duration (float): Optional duration to analyze (None means full file)
        
    Returns:
        list: List of beat timestamps in seconds
        list: Beat strengths (onset envelope values at beat times)
    """
    print(f"Loading audio file: {audio_file}")
    
    # Load the full audio file by setting duration parameter
    y, sr = librosa.load(audio_file, sr=None, duration=duration)
    
    print(f"Audio loaded: {len(y)/sr:.2f} seconds at {sr}Hz")
    
    # Get onset envelope with larger frame length for better beat detection
    hop_length = 512  # Default hop size
    onset_env = librosa.onset.onset_strength(
        y=y, 
        sr=sr,
        hop_length=hop_length,
        aggregate=np.median  # More robust aggregation
    )
    
    # Dynamic beats with appropriate tempo constraints
    tempo, beats = librosa.beat.beat_track(
        onset_envelope=onset_env, 
        sr=sr,
        hop_length=hop_length,
        start_bpm=min_bpm,
        tightness=100
    )
    
    # Handle tempo being an array or scalar
    if isinstance(tempo, np.ndarray):
        tempo = float(tempo[0])
        
    print(f"Detected tempo: {tempo:.1f} BPM")
    
    # If we're getting too few beats with beat_track, try onset detection
    if len(beats) < 10 and len(y)/sr > 20:  # If less than 10 beats for > 20s audio
        print("Few beats detected, trying onset detection method...")
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=sr,
            hop_length=hop_length,
            backtrack=True
        )
        beats = onset_frames  # Use onsets as beats
    
    # Convert frame indices to time
    beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=hop_length)
    
    # Get onset strengths at beat times
    beat_strengths = []
    for beat in beats:
        if beat < len(onset_env):
            strength = onset_env[beat]
            beat_strengths.append(strength)
        else:
            beat_strengths.append(0)
    
    # Normalize strengths to range [0,1]
    if beat_strengths:
        min_strength = min(beat_strengths)
        max_strength = max(beat_strengths)
        range_strength = max_strength - min_strength
        
        if range_strength > 0:
            beat_strengths = [(s - min_strength) / range_strength for s in beat_strengths]
        else:
            beat_strengths = [0.5 for _ in beat_strengths]
    
    # Convert frame indices to time
    beat_times = librosa.frames_to_time(beats, sr=sr)
    
    # Get onset strengths at beat times
    beat_strengths = []
    for beat in beats:
        if beat < len(onset_env):
            strength = onset_env[beat]
            beat_strengths.append(strength)
        else:
            beat_strengths.append(0)
    
    # Normalize strengths to range [0,1]
    if beat_strengths:
        min_strength = min(beat_strengths)
        max_strength = max(beat_strengths)
        range_strength = max_strength - min_strength
        
        if range_strength > 0:
            beat_strengths = [(s - min_strength) / range_strength for s in beat_strengths]
        else:
            beat_strengths = [0.5 for _ in beat_strengths]
    
    # Filter beats by threshold
    filtered_beats = []
    filtered_strengths = []
    for time, strength in zip(beat_times, beat_strengths):
        if strength >= threshold:
            filtered_beats.append(time)
            filtered_strengths.append(strength)
    
    print(f"Detected {len(filtered_beats)} beats above threshold {threshold}")
    
    return filtered_beats, filtered_strengths

def save_beats_to_csv(beat_times, beat_strengths, output_file):
    """
    Save beat timestamps and strengths to a CSV file.
    
    Args:
        beat_times (list): List of beat timestamps in seconds
        beat_strengths (list): List of beat strengths (0-1)
        output_file (str): Path to the output CSV file
    """
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['beat_number', 'timestamp', 'strength'])
        
        for i, (time, strength) in enumerate(zip(beat_times, beat_strengths)):
            writer.writerow([i+1, f"{time:.6f}", f"{strength:.6f}"])
    
    print(f"Beat information saved to {output_file}")


def extract_rhythmic_features(audio_file, duration=None):
    """
    Extract rhythmic features from an audio file.
    
    Args:
        audio_file (str): Path to the audio file
        duration (float): Optional duration to analyze (None means full file)
        
    Returns:
        dict: Dictionary containing tempogram, fourier_tempogram, and tempogram_ratio features
    """
    print(f"Loading audio file for rhythmic analysis: {audio_file}")
    
    # Load the audio file
    y, sr = librosa.load(audio_file, sr=None, duration=duration)
    
    # Compute onset envelope
    hop_length = 512  # Default hop size
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    
    # Compute tempogram
    tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
    
    # Compute fourier tempogram
    fourier_tempogram = librosa.feature.fourier_tempogram(y=y, sr=sr, hop_length=hop_length)
    
    # Compute tempogram ratio
    tempogram_ratio = librosa.feature.tempogram_ratio(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
    
    # Get timestamps for each frame
    frame_times = librosa.frames_to_time(np.arange(tempogram.shape[1]), sr=sr, hop_length=hop_length)
    
    return {
        'frame_times': frame_times,
        'tempogram': tempogram,
        'fourier_tempogram': fourier_tempogram,
        'tempogram_ratio': tempogram_ratio
    }

def save_tempogram_to_csv(frame_times, tempogram, output_file):
    """
    Save tempogram data to a CSV file.
    
    Args:
        frame_times (array): Array of frame timestamps
        tempogram (array): Tempogram feature matrix
        output_file (str): Path to the output CSV file
    """
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Create header with tempo bins
        header = ['timestamp'] + [f'tempo_bin_{i}' for i in range(tempogram.shape[0])]
        writer.writerow(header)
        
        # Write each frame's data
        for i, time in enumerate(frame_times):
            if i < tempogram.shape[1]:
                row = [f"{time:.6f}"] + [f"{val:.6f}" for val in tempogram[:, i]]
                writer.writerow(row)
    
    print(f"Tempogram data saved to {output_file}")

def save_fourier_tempogram_to_csv(frame_times, fourier_tempogram, output_file):
    """
    Save Fourier tempogram data to a CSV file.
    
    Args:
        frame_times (array): Array of frame timestamps
        fourier_tempogram (array): Fourier tempogram feature matrix
        output_file (str): Path to the output CSV file
    """
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Create header with frequency bins
        header = ['timestamp'] + [f'freq_bin_{i}' for i in range(fourier_tempogram.shape[0])]
        writer.writerow(header)
        
        # Write each frame's data
        for i, time in enumerate(frame_times):
            if i < fourier_tempogram.shape[1]:
                # Convert complex values to magnitudes
                magnitudes = np.abs(fourier_tempogram[:, i])
                row = [f"{time:.6f}"] + [f"{val:.6f}" for val in magnitudes]
                writer.writerow(row)
    
    print(f"Fourier tempogram data saved to {output_file}")

def save_tempogram_ratio_to_csv(frame_times, tempogram_ratio, output_file):
    """
    Save tempogram ratio data to a CSV file.
    
    Args:
        frame_times (array): Array of frame timestamps
        tempogram_ratio (array): Tempogram ratio feature matrix
        output_file (str): Path to the output CSV file
    """
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Create header with ratio bins
        header = ['timestamp'] + [f'ratio_bin_{i}' for i in range(tempogram_ratio.shape[0])]
        writer.writerow(header)
        
        # Write each frame's data
        for i, time in enumerate(frame_times):
            if i < tempogram_ratio.shape[1]:
                row = [f"{time:.6f}"] + [f"{val:.6f}" for val in tempogram_ratio[:, i]]
                writer.writerow(row)
    
    print(f"Tempogram ratio data saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Extract beats and rhythmic features from audio file')
    parser.add_argument('--input', required=True, help='Input audio file path')
    parser.add_argument('--output', required=True, help='Output CSV file base path (without extension)')
    parser.add_argument('--threshold', type=float, default=0.2, 
                        help='Onset strength threshold (0.0-1.0)')
    parser.add_argument('--min_bpm', type=int, default=60, 
                        help='Minimum tempo in BPM')
    parser.add_argument('--max_bpm', type=int, default=180, 
                        help='Maximum tempo in BPM')
    parser.add_argument('--duration', type=float, default=None,
                        help='Duration in seconds to analyze (default: entire file)')
    parser.add_argument('--extract_rhythmic', action='store_true',
                        help='Extract additional rhythmic features (tempogram, etc.)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found")
        return
    
    # Extract and save beats
    beats_output = args.output if args.output.endswith('.csv') else f"{args.output}_beats.csv"
    beat_times, beat_strengths = extract_beats(
        args.input, 
        threshold=args.threshold,
        min_bpm=args.min_bpm,
        max_bpm=args.max_bpm,
        duration=args.duration
    )
    
    save_beats_to_csv(beat_times, beat_strengths, beats_output)
    
    print(f"Beat extraction complete - {len(beat_times)} beats detected")
    
    # Print some diagnostic info
    if beat_times:
        avg_bpm = 60 * (len(beat_times) - 1) / (beat_times[-1] - beat_times[0])
        print(f"Average BPM: {avg_bpm:.1f}")
        print(f"Audio duration: {beat_times[-1]:.2f} seconds")
    
    # Extract and save rhythmic features if requested
    if args.extract_rhythmic:
        print("Extracting rhythmic features...")
        
        # Create output file paths
        base_output = args.output.replace('.csv', '')
        tempogram_output = f"{base_output}_tempogram.csv"
        fourier_output = f"{base_output}_fourier_tempogram.csv"
        ratio_output = f"{base_output}_tempogram_ratio.csv"
        
        # Extract features
        features = extract_rhythmic_features(args.input, duration=args.duration)
        
        # Save features to CSV files
        save_tempogram_to_csv(features['frame_times'], features['tempogram'], tempogram_output)
        save_fourier_tempogram_to_csv(features['frame_times'], features['fourier_tempogram'], fourier_output)
        save_tempogram_ratio_to_csv(features['frame_times'], features['tempogram_ratio'], ratio_output)
        
        print("Rhythmic feature extraction complete")


if __name__ == "__main__":
    main()