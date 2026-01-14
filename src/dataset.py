import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path

from . import config
from .features import load_processed_data, process_dataset


class MusicMFCCDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.FloatTensor(features.reshape(features.shape[0], -1))
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class MusicSpectrogramDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.FloatTensor(features).unsqueeze(1)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def get_data_loaders(feature_type: str = 'mfcc',
                     batch_size: int = 32,
                     train_split: float = 0.8,
                     process_if_missing: bool = True) -> tuple:
    try:
        features, labels, genres = load_processed_data(feature_type)
    except FileNotFoundError:
        if process_if_missing:
            print(f"Processed data not found. Processing {feature_type} features...")
            features, labels, _ = process_dataset(feature_type=feature_type)
            _, labels, genres = load_processed_data(feature_type)
        else:
            raise
    
    np.random.seed(config.RANDOM_SEED)
    indices = np.random.permutation(len(labels))
    split_idx = int(len(indices) * train_split)
    
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    train_features = features[train_indices]
    train_labels = labels[train_indices]
    test_features = features[test_indices]
    test_labels = labels[test_indices]
    
    if feature_type == 'mfcc':
        train_dataset = MusicMFCCDataset(train_features, train_labels)
        test_dataset = MusicMFCCDataset(test_features, test_labels)
        num_features = train_dataset.features.shape[1]
    else:
        train_dataset = MusicSpectrogramDataset(train_features, train_labels)
        test_dataset = MusicSpectrogramDataset(test_features, test_labels)
        num_features = train_dataset.features.shape[1:]
    
    effective_batch_size = min(batch_size, len(train_dataset))
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=effective_batch_size, 
        shuffle=True,
        drop_last=len(train_dataset) > effective_batch_size
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=min(batch_size, len(test_dataset)) if len(test_dataset) > 0 else 1, 
        shuffle=False
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Feature shape: {num_features}")
    
    return train_loader, test_loader, num_features, genres


def prepare_data_for_clustering(model, data_loader, device='cuda'):
    model.eval()
    latent_features = []
    all_labels = []
    
    with torch.no_grad():
        for batch_features, batch_labels in data_loader:
            batch_features = batch_features.to(device)
            mu, _ = model.encode(batch_features)
            latent_features.append(mu.cpu().numpy())
            all_labels.append(batch_labels.numpy())
    
    latent_features = np.concatenate(latent_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    return latent_features, all_labels
