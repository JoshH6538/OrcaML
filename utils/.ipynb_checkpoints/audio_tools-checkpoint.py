import os
import soundfile as sf
import librosa
import numpy as np

def extract_clip_and_mfcc(wav_path, start_time, duration, output_path, n_mfcc=13):
    """
    Extracts a clip from a WAV file, saves it, and returns average MFCCs and MFCC deltas.
    """
    try:
        data, sr = sf.read(wav_path)
        start_sample = int(start_time * sr)
        end_sample = int((start_time + duration) * sr)
        clip = data[start_sample:end_sample]

        if len(data) == 0 or (end_sample - start_sample) < 2048:
            print("Skipping: clip too short or empty")
            return None, None


        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sf.write(output_path, clip, sr)

        # Extract MFCCs
        mfcc = librosa.feature.mfcc(y=clip, sr=sr, n_mfcc=n_mfcc)

        # Compute first-order delta
        mfcc_delta = librosa.feature.delta(mfcc)

        # Average over time
        mfcc_mean = np.mean(mfcc, axis=1)
        delta_mean = np.mean(mfcc_delta, axis=1)

        return mfcc_mean, delta_mean

    except Exception as e:
        print(f"Error extracting clip: {e}")
        return None, None

import random

def extract_negative_clip_and_mfcc(wav_path, exclude_ranges, output_path, duration=2.0, n_mfcc=13, max_attempts=10):
    """
    Extracts a random negative clip from a WAV file (avoiding orca call ranges),
    saves it, and returns average MFCCs and MFCC deltas.
    """
    try:
        data, sr = sf.read(wav_path)
        total_duration = len(data) / sr

        for _ in range(max_attempts):
            start = random.uniform(0, total_duration - duration)
            end = start + duration

            # Check for overlap with any excluded range
            overlaps = any(start < r_end and end > r_start for r_start, r_end in exclude_ranges)
            if not overlaps:
                start_sample = int(start * sr)
                end_sample = int(end * sr)
                clip = data[start_sample:end_sample]

                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                sf.write(output_path, clip, sr)

                mfcc = librosa.feature.mfcc(y=clip, sr=sr, n_mfcc=n_mfcc)
                mfcc_delta = librosa.feature.delta(mfcc)

                mfcc_mean = np.mean(mfcc, axis=1)
                delta_mean = np.mean(mfcc_delta, axis=1)

                return mfcc_mean, delta_mean

        return None, None

    except Exception as e:
        print(f"Error extracting negative clip: {e}")
        return None, None

def extract_positive_clip(wav_path, start_time, duration, output_path, n_mfcc=13):
    """
    Extracts a clip from a WAV file, saves it, and returns average MFCCs and MFCC deltas.
    """
    try:
        # obtain clip segment
        data, sr = sf.read(wav_path)
        start_sample = int(start_time * sr)
        end_sample = int((start_time + duration) * sr)
        clip = data[start_sample:end_sample]

        # Convert minimum duration to samples
        min_length_sec = 0.3
        min_samples = int(min_length_sec * sr)
        
        if len(clip) < min_samples:
            print(f"Skipping clip: too short ({len(clip)/sr:.2f}s)")
            return None, None


        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sf.write(output_path, clip, sr)

        # Extract MFCCs
        mfcc = librosa.feature.mfcc(y=clip, sr=sr, n_mfcc=n_mfcc)

        # Compute first-order delta
        mfcc_delta = librosa.feature.delta(mfcc)

        # Average over time
        mfcc_mean = np.mean(mfcc, axis=1)
        delta_mean = np.mean(mfcc_delta, axis=1)

        return mfcc_mean, delta_mean

    except Exception as e:
        print(f"Error extracting clip: {e}")
        return None, None

def extract_negative_clip(wav_path, exclude_ranges, clip_index, output_dir, duration=2.0, buffer=1.0, n_mfcc=13):
    import soundfile as sf
    import os
    import numpy as np
    import random
    import librosa

    wav_base = os.path.splitext(os.path.basename(wav_path))[0]

    try:
        data, sr = sf.read(wav_path)
        total_duration = len(data) / sr
    except Exception as e:
        print(f"Error reading {wav_path}: {e}")
        return None, None

    for _ in range(10):  # Try 10 times to find a valid clip
        start = random.uniform(0, total_duration - duration)
        end = start + duration

        # Skip overlapping regions
        overlaps = any(start < r_end and end > r_start for r_start, r_end in exclude_ranges)
        if overlaps:
            continue

        # Extract audio
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        clip = data[start_sample:end_sample]

        if len(clip) < int(0.3 * sr):
            continue

        # Generate filename
        start_str = f"{start:.2f}".replace(".", "_")
        filename = f"neg_{clip_index:05}_{wav_base}_{start_str}.wav"
        output_path = os.path.join(output_dir, filename)

        # Save file
        sf.write(output_path, clip, sr)

        # Extract MFCC and delta features
        try:
            mfcc = librosa.feature.mfcc(y=clip, sr=sr, n_mfcc=n_mfcc)
            delta = librosa.feature.delta(mfcc)
            mfcc_mean = np.mean(mfcc, axis=1)
            delta_mean = np.mean(delta, axis=1)
            return mfcc_mean, delta_mean
        except Exception as e:
            print(f"Error extracting MFCCs: {e}")
            return None, None

    # print(f"⚠️ No valid negative segment found in {wav_path}")
    return None, None


def extract_negatives_iterative(
    wav_filename,
    annotations_df,
    output_dir,
    clip_start_index=1,
    duration=2.0,
    buffer=1.0,
    n_mfcc=13,
    max_clips=None
):
    import os
    import soundfile as sf
    import numpy as np
    import librosa

    records = []
    clip_index = clip_start_index

    wav_path = os.path.join("data", "wav", wav_filename)
    wav_base = os.path.splitext(wav_filename)[0]

    try:
        data, sr = sf.read(wav_path)
    except Exception as e:
        print(f"❌ Could not read {wav_path}: {e}")
        return records

    total_duration = len(data) / sr

    # Get all call times for this file
    file_annotations = annotations_df[annotations_df["wav_filename"] == wav_filename]
    exclude_ranges = [
        (max(0, row["start_time_s"] - buffer), row["start_time_s"] + row["duration_s"] + buffer)
        for _, row in file_annotations.iterrows()
    ]

    # Step through the file in 2s increments
    t = 0.0
    while t + duration <= total_duration:
        start = t
        end = t + duration
        t += duration

        # Skip overlaps
        if any(start < r_end and end > r_start for r_start, r_end in exclude_ranges):
            continue

        start_sample = int(start * sr)
        end_sample = int(end * sr)
        clip = data[start_sample:end_sample]

        if len(clip) < int(0.3 * sr):  # Skip very short clips
            continue

        # Extract features
        try:
            mfcc = librosa.feature.mfcc(y=clip, sr=sr, n_mfcc=n_mfcc)
            delta = librosa.feature.delta(mfcc)
            mfcc_mean = np.mean(mfcc, axis=1)
            delta_mean = np.mean(delta, axis=1)
        except Exception as e:
            print(f"⚠️ Failed to extract features for t={start:.2f}: {e}")
            continue

        # Save clip
        start_str = f"{start:.2f}".replace(".", "_")
        filename = f"neg_{clip_index:05}_{wav_base}_{start_str}.wav"
        output_path = os.path.join(output_dir, filename)
        sf.write(output_path, clip, sr)

        # Build record
        record = {
            "clip_name": filename,
            "label": "no_call",
            "source_wav": wav_filename,
            "start_time": start
        }
        for i, val in enumerate(mfcc_mean):
            record[f"mfcc_{i+1}"] = val
        for i, val in enumerate(delta_mean):
            record[f"delta_mfcc_{i+1}"] = val

        records.append(record)
        clip_index += 1

        if max_clips and len(records) >= max_clips:
            break

    print(f"Extracted {len(records)} negative clips from {wav_filename}")
    return records


    