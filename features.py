import numpy as np
import librosa
import os
from pathlib import Path
from tqdm import tqdm
import pickle

from . import config


def load_audio(file_path: str, sr: int = config.SAMPLE_RATE, 
               duration: float = config.DURATION) -> np.ndarray:
    try:
        y, _ = librosa.load(file_path, sr=sr, duration=duration)
        target_length = int(sr * duration)
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)), mode='constant')
        elif len(y) > target_length:
            y = y[:target_length]
        return y
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def extract_mfcc(y: np.ndarray, sr: int = config.SAMPLE_RATE,
                 n_mfcc: int = config.N_MFCC,
                 hop_length: int = config.HOP_LENGTH) -> np.ndarray:
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
    return mfcc


def extract_mel_spectrogram(y: np.ndarray, sr: int = config.SAMPLE_RATE,
                            n_mels: int = config.N_MELS,
                            hop_length: int = config.HOP_LENGTH,
                            n_fft: int = config.N_FFT) -> np.ndarray:
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, hop_length=hop_length, n_fft=n_fft
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
    return mel_spec_norm


def extract_combined_features(y: np.ndarray, sr: int = config.SAMPLE_RATE) -> dict:
    features = {
        'mfcc': extract_mfcc(y, sr),
        'mel_spectrogram': extract_mel_spectrogram(y, sr),
        'chroma': librosa.feature.chroma_stft(y=y, sr=sr, hop_length=config.HOP_LENGTH),
        'spectral_contrast': librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=config.HOP_LENGTH),
    }
    return features


def process_dataset(data_dir: str = str(config.RAW_DATA_DIR),
                   genres: list = config.GENRES,
                   feature_type: str = 'mfcc',
                   save_dir: str = str(config.PROCESSED_DATA_DIR)) -> tuple:
    features_list = []
    labels = []
    file_paths = []
    
    data_path = Path(data_dir)
    
    print(f"Processing dataset from: {data_path}")
    print(f"Genres: {genres}")
    print(f"Feature type: {feature_type}")
    
    for genre_idx, genre in enumerate(genres):
        genre_path = data_path / genre
        
        if not genre_path.exists():
            print(f"Warning: Genre folder not found: {genre_path}")
            continue
        
        audio_extensions = ['.wav', '.mp3', '.au', '.flac']
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(list(genre_path.glob(f"*{ext}")))
        
        print(f"\nProcessing {genre}: {len(audio_files)} files")
        
        for audio_file in tqdm(audio_files, desc=genre):
            y = load_audio(str(audio_file))
            if y is None:
                continue
            
            if feature_type == 'mfcc':
                feat = extract_mfcc(y)
            elif feature_type == 'mel_spectrogram':
                feat = extract_mel_spectrogram(y)
            elif feature_type == 'combined':
                feat = extract_combined_features(y)
            else:
                raise ValueError(f"Unknown feature type: {feature_type}")
            
            features_list.append(feat)
            labels.append(genre_idx)
            file_paths.append(str(audio_file))
    
    if feature_type != 'combined':
        features = np.array(features_list, dtype=np.float32)
    else:
        features = features_list
    
    labels = np.array(labels, dtype=np.int64)
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    data = {
        'features': features,
        'labels': labels,
        'file_paths': file_paths,
        'genres': genres,
        'feature_type': feature_type
    }
    
    save_file = save_path / f"processed_{feature_type}.pkl"
    with open(save_file, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"\nSaved processed data to: {save_file}")
    print(f"Total samples: {len(labels)}")
    print(f"Samples per genre: {np.bincount(labels)}")
    
    return features, labels, file_paths


def load_processed_data(feature_type: str = 'mfcc',
                        data_dir: str = str(config.PROCESSED_DATA_DIR)) -> tuple:
    load_path = Path(data_dir) / f"processed_{feature_type}.pkl"
    
    if not load_path.exists():
        raise FileNotFoundError(f"Processed data not found: {load_path}")
    
    with open(load_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Loaded {len(data['labels'])} samples")
    return data['features'], data['labels'], data['genres']


if __name__ == "__main__":
    print("Processing MFCC features...")
    process_dataset(feature_type='mfcc')
    
    print("\nProcessing Mel-Spectrogram features...")
    process_dataset(feature_type='mel_spectrogram')
