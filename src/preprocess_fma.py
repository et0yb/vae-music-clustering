import numpy as np
import librosa
from pathlib import Path
import json
from tqdm import tqdm
import os

DATA_DIR = Path('data/fma_english')
OUTPUT_DIR = Path('data/processed')
SAMPLE_RATE = 22050
DURATION = 30
N_MFCC = 40
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048
MAX_SAMPLES_PER_GENRE = 50


def extract_features(audio_path, feature_type='mfcc'):
    try:
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=DURATION)
        
        if len(y) < SAMPLE_RATE * 5:
            return None, None
        
        target_length = SAMPLE_RATE * DURATION
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)))
        else:
            y = y[:target_length]
        
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH, n_fft=N_FFT)
        mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)
        
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH, n_fft=N_FFT)
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min() + 1e-8)
        
        return mfcc, mel_spec
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None, None


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    genres = [d.name for d in DATA_DIR.iterdir() if d.is_dir()]
    print(f"Found genres: {genres}")
    
    genre_to_idx = {g: i for i, g in enumerate(sorted(genres))}
    idx_to_genre = {i: g for g, i in genre_to_idx.items()}
    
    all_mfccs = []
    all_mel_specs = []
    all_genre_labels = []
    all_metadata = []
    
    for genre in tqdm(genres, desc="Processing genres"):
        genre_dir = DATA_DIR / genre
        audio_dir = genre_dir / "audio"
        
        if not audio_dir.exists():
            audio_dir = genre_dir
        
        audio_files = list(audio_dir.glob("*.mp3")) + list(audio_dir.glob("*.wav"))
        
        count = 0
        for audio_file in tqdm(audio_files, desc=f"  {genre}", leave=False):
            if count >= MAX_SAMPLES_PER_GENRE:
                break
            
            mfcc, mel_spec = extract_features(audio_file)
            
            if mfcc is not None:
                all_mfccs.append(mfcc)
                all_mel_specs.append(mel_spec)
                all_genre_labels.append(genre_to_idx[genre])
                all_metadata.append({
                    'file': str(audio_file),
                    'genre': genre,
                    'genre_idx': genre_to_idx[genre]
                })
                count += 1
        
        print(f"  {genre}: {count} samples")
    
    mfccs = np.array(all_mfccs)
    mel_specs = np.array(all_mel_specs)
    genre_labels = np.array(all_genre_labels)
    language_labels = np.zeros_like(genre_labels)  # All English = 0
    
    mfcc_flat = mfccs.reshape(mfccs.shape[0], -1)
    mel_flat = mel_specs.reshape(mel_specs.shape[0], -1)
    combined = np.concatenate([mfcc_flat, mel_flat], axis=1)
    
    np.save(OUTPUT_DIR / 'mfccs.npy', mfccs)
    np.save(OUTPUT_DIR / 'mfcc_flat.npy', mfcc_flat)
    np.save(OUTPUT_DIR / 'mel_spectrograms.npy', mel_specs)
    np.save(OUTPUT_DIR / 'combined_features.npy', combined)
    np.save(OUTPUT_DIR / 'genre_labels.npy', genre_labels)
    np.save(OUTPUT_DIR / 'language_labels.npy', language_labels)
    
    metadata = {
        'num_samples': len(genre_labels),
        'num_genres': len(genres),
        'genre_to_idx': genre_to_idx,
        'idx_to_genre': idx_to_genre,
        'mfcc_shape': list(mfccs.shape),
        'mel_shape': list(mel_specs.shape),
        'combined_shape': list(combined.shape)
    }
    
    with open(OUTPUT_DIR / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    with open(OUTPUT_DIR / 'song_info.json', 'w') as f:
        json.dump(all_metadata, f, indent=2)
    
    print(f"\nDataset Summary:")
    print(f"  Total samples: {len(genre_labels)}")
    print(f"  MFCC shape: {mfccs.shape}")
    print(f"  Mel spectrogram shape: {mel_specs.shape}")
    print(f"  Combined features shape: {combined.shape}")
    print(f"  Genres: {list(genre_to_idx.keys())}")
    print(f"\nSaved to {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
