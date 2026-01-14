# Music Genre Clustering Using VAEs

CSE425 Neural Network Course Project - BRAC University

## About This Project

This project uses Variational Autoencoders (VAE) to cluster music tracks by genre. I use the FMA (Free Music Archive) dataset which has songs from different genres like rock, pop, electronic, folk etc.

The project has 3 difficulty levels:
- **Easy Task**: Basic VAE with MFCC features + K-Means clustering
- **Medium Task**: Convolutional VAE with Mel-Spectrograms + multiple clustering methods
- **Hard Task**: Beta-VAE with disentanglement for better representations

## Project Structure

```
project/
├── src/
│   ├── easy_task.py      # Easy task - run this for basic VAE
│   ├── medium_task.py    # Medium task - Conv-VAE 
│   ├── hard_task.py      # Hard task - Beta-VAE
│   ├── preprocess_fma.py # Preprocess FMA dataset
│   ├── config.py         # All configurations
│   ├── vae.py            # Basic VAE model
│   ├── conv_vae.py       # Convolutional VAE
│   ├── beta_vae.py       # Beta-VAE for disentanglement
│   ├── clustering.py     # K-Means, DBSCAN, Agglomerative
│   ├── evaluation.py     # Silhouette, ARI, NMI, Purity etc
│   ├── visualization.py  # t-SNE, UMAP plots
│   ├── features.py       # MFCC and Mel-Spectrogram extraction
│   └── dataset.py        # PyTorch dataset classes
├── data/
│   ├── fma_english/      # FMA audio files by genre
│   └── processed/        # Extracted features (created by preprocess)
├── results/              # Output plots and metrics saved here
├── report.tex            # LaTeX report
└── requirements.txt
```

## How to Run

### 1. Install Requirements

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

Put FMA audio files in `data/fma_english/` folder organized by genre:
```
data/fma_english/
├── electronic/
│   ├── song1.mp3
│   └── song2.mp3
├── rock/
├── pop/
└── folk/
```

### 3. Preprocess Dataset

```bash
cd src
python preprocess_fma.py
```

### 4. Run the Tasks

For Easy Task (Basic VAE + K-Means):
```bash
python easy_task.py
```

For Medium Task (Conv-VAE + Multiple Clustering):
```bash
python medium_task.py
```

For Hard Task (Beta-VAE):
```bash
python hard_task.py
```

## Features Used

- **MFCC**: 40 coefficients 
- **Mel-Spectrogram**: 128 mel bands

## Clustering Methods

- K-Means
- Gaussian Mixture Model (GMM)
- Agglomerative Clustering (Ward linkage)
- Spectral Clustering

## Evaluation Metrics

- Silhouette Score
- Calinski-Harabasz Index
- Davies-Bouldin Index
- Adjusted Rand Index (ARI)
- Normalized Mutual Information (NMI)
- Cluster Purity

## Results

Results are saved in the `results/` folder including:
- t-SNE and UMAP visualizations
- Clustering comparison plots
- Metric tables
- Trained models (.pth files)

## Requirements

- Python 3.8+
- PyTorch
- librosa
- scikit-learn
- matplotlib
- umap-learn

See `requirements.txt` for full list.

## Note

The dataset files (audio) are not included in this repo due to size. You need to download FMA dataset and add audio files in the `data/fma_english/` folder.

---
Nayem Bin Omar  
BRAC University
