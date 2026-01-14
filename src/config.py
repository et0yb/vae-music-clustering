import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

GENRES = ["rabindra_sangeet", "nazrul_geeti", "adhunik"]

def get_genres_from_data():
    if RAW_DATA_DIR.exists():
        folders = [d.name for d in RAW_DATA_DIR.iterdir() if d.is_dir()]
        if folders:
            return sorted(folders)
    return GENRES

GENRES = get_genres_from_data()
NUM_CLASSES = len(GENRES)

SAMPLE_RATE = 22050
DURATION = 10
N_MFCC = 40
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048
SAMPLES_PER_CLIP = SAMPLE_RATE * DURATION

EASY_VAE_CONFIG = {
    "input_dim": N_MFCC * (SAMPLES_PER_CLIP // HOP_LENGTH + 1),
    "hidden_dims": [512, 256, 128],
    "latent_dim": 32,
    "learning_rate": 1e-3,
    "batch_size": 8,
    "epochs": 100,
}

MEDIUM_VAE_CONFIG = {
    "in_channels": 1,
    "latent_dim": 64,
    "hidden_channels": [32, 64, 128, 256],
    "learning_rate": 1e-3,
    "batch_size": 8,
    "epochs": 150,
}

HARD_VAE_CONFIG = {
    "in_channels": 1,
    "latent_dim": 64,
    "hidden_channels": [32, 64, 128, 256],
    "beta": 4.0,
    "learning_rate": 1e-4,
    "batch_size": 8,
    "epochs": 200,
}

CLUSTERING_CONFIG = {
    "n_clusters": NUM_CLASSES,
    "random_state": 42,
    "dbscan_eps": 0.5,
    "dbscan_min_samples": 5,
    "linkage": "ward",
}

DEVICE = "cuda"
RANDOM_SEED = 42
TRAIN_TEST_SPLIT = 0.8

VIS_CONFIG = {
    "tsne_perplexity": 30,
    "tsne_n_iter": 1000,
    "umap_n_neighbors": 15,
    "umap_min_dist": 0.1,
    "figure_size": (10, 8),
}
