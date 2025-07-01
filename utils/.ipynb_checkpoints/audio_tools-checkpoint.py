import os
import soundfile as sf
import librosa
import numpy as np
import random

def extract_clip_and_mfcc(wav_path, start_time, duration, output_path, n_mfcc=13):
    """
    Extracts a segment from a WAV file, saves it, and returns average MFCCs.
    """
    try:
        data, samplerate = sf.read(wav_path)
        start_sample = int(start_time * samplerate)
        end_sample = int((start_time + duration) * samplerate)
        clip = data[start_sample:end_sample]

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sf.write(output_path, clip, samplerate)

        mfcc = librosa.feature.mfcc(y=clip, sr=samplerate, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc, axis=1)

        return mfcc_mean

    except Exception as e:
        print(f"Error extracting clip: {e}")

def extract_negative_clip_and_mfcc(wav_path, exclude_ranges, output_path, duration=2.0, n_mfcc=13, max_attempts=10):
    """
    Extracts a random negative clip from a WAV file (avoiding orca call ranges),
    saves it, and returns average MFCCs.

    Parameters:
        wav_path (str): Path to source WAV
        exclude_ranges (list of (start, end)): Time ranges to avoid (calls)
        output_path (str): Path to save the clip
        duration (float): Duration of the clip in seconds
        n_mfcc (int): Number of MFCCs to return
        max_attempts (int): How many times to try picking a valid clip

    Returns:
        np.ndarray: 1D array of average MFCCs
    """
    try:
        data, sr = sf.read(wav_path)
        total_duration = len(data) / sr

        for _ in range(max_attempts):
            start = random.uniform(0, total_duration - duration)
            end = start + duration

            # Check for overlap with any excluded ranges
            overlaps = any(start < r_end and end > r_start for r_start, r_end in exclude_ranges)
            if not overlaps:
                # Extract and save
                start_sample = int(start * sr)
                end_sample = int(end * sr)
                clip = data[start_sample:end_sample]

                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                sf.write(output_path, clip, sr)

                mfcc = librosa.feature.mfcc(y=clip, sr=sr, n_mfcc=n_mfcc)
                mfcc_mean = np.mean(mfcc, axis=1)
                return mfcc_mean

        return None

    except Exception as e:
        print(f"Error extracting negative clip: {e}")
        return None