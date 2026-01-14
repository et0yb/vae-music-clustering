import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    adjusted_rand_score, normalized_mutual_info_score, confusion_matrix
)
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

CONFIG = {
    'latent_dim': 64,
    'epochs': 150,
    'batch_size': 32,
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
    'beta': 1.0,
    'data_dir': 'data/processed',
    'results_dir': 'results/medium'
}


class ConvVAE(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )
        
        self.flatten_size = 256 * 8 * 82
        
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)
        
        self.fc_decode = nn.Linear(latent_dim, self.flatten_size)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(h.size(0), 256, 8, 82)
        return self.decoder(h)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z


def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


def load_data(data_dir):
    print("\nLoading mel spectrograms...")
    
    mel_specs = np.load(f'{data_dir}/mel_spectrograms.npy')
    genre_labels = np.load(f'{data_dir}/genre_labels.npy')
    
    with open(f'{data_dir}/metadata.json', 'r') as f:
        metadata = json.load(f)
    
    print(f"   Shape: {mel_specs.shape}")
    
    mel_specs = (mel_specs - mel_specs.min()) / (mel_specs.max() - mel_specs.min() + 1e-8)
    mel_specs = mel_specs[:, np.newaxis, :, :]
    
    print(f"   Normalized shape: {mel_specs.shape}")
    
    return mel_specs, genre_labels, metadata


def train_epoch(model, loader, optimizer, scheduler, beta):
    model.train()
    total_loss, total_recon, total_kl = 0, 0, 0
    
    for data, in loader:
        data = data.to(DEVICE)
        optimizer.zero_grad()
        
        recon, mu, logvar, z = model(data)
        loss, recon_loss, kl_loss = vae_loss(recon, data, mu, logvar, beta)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()
    
    n = len(loader.dataset)
    return total_loss / n, total_recon / n, total_kl / n


def extract_latent(model, data_tensor):
    model.eval()
    latents = []
    
    with torch.no_grad():
        loader = DataLoader(TensorDataset(data_tensor), batch_size=32)
        for batch, in loader:
            batch = batch.to(DEVICE)
            mu, _ = model.encode(batch)
            latents.append(mu.cpu().numpy())
    
    return np.vstack(latents)


def evaluate_clustering(latent, true_labels, n_clusters):
    results = {}
    
    methods = {
        'K-Means': KMeans(n_clusters=n_clusters, random_state=42, n_init=20),
        'GMM': GaussianMixture(n_components=n_clusters, random_state=42),
        'Agglomerative': AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    }
    
    best_method, best_ari = None, -1
    
    for name, clusterer in methods.items():
        try:
            pred = clusterer.fit_predict(latent)
            
            sil = silhouette_score(latent, pred)
            ch = calinski_harabasz_score(latent, pred)
            db = davies_bouldin_score(latent, pred)
            ari = adjusted_rand_score(true_labels, pred)
            nmi = normalized_mutual_info_score(true_labels, pred)
            
            contingency = confusion_matrix(true_labels, pred)
            purity = np.sum(np.max(contingency, axis=0)) / len(true_labels)
            
            results[name] = {
                'silhouette': sil,
                'calinski_harabasz': ch,
                'davies_bouldin': db,
                'ari': ari,
                'nmi': nmi,
                'purity': purity,
                'predictions': pred
            }
            
            if ari > best_ari:
                best_ari = ari
                best_method = name
                
        except Exception as e:
            print(f"   {name} failed: {e}")
    
    return results, best_method


def visualize_results(latent, genre_labels, results, best_method, results_dir, metadata, data, model):
    os.makedirs(results_dir, exist_ok=True)
    
    n_genres = metadata['num_genres']
    idx_to_genre = metadata['idx_to_genre']
    genre_names = [idx_to_genre[str(i)] for i in range(n_genres)]
    
    print("   Computing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    latent_2d = tsne.fit_transform(latent)
    
    print("   Computing UMAP...")
    reducer = umap.UMAP(n_components=2, random_state=42)
    latent_umap = reducer.fit_transform(latent)
    
    palette = sns.color_palette("husl", n_genres)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    ax = axes[0, 0]
    for i in range(n_genres):
        mask = genre_labels == i
        ax.scatter(latent_2d[mask, 0], latent_2d[mask, 1], 
                   c=[palette[i]], label=genre_names[i], alpha=0.7)
    ax.set_title('t-SNE: Ground Truth')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    
    ax = axes[0, 1]
    for i in range(n_genres):
        mask = genre_labels == i
        ax.scatter(latent_umap[mask, 0], latent_umap[mask, 1], 
                   c=[palette[i]], label=genre_names[i], alpha=0.7)
    ax.set_title('UMAP: Ground Truth')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    
    pred = results[best_method]['predictions']
    ax = axes[1, 0]
    for i in range(n_genres):
        mask = pred == i
        ax.scatter(latent_2d[mask, 0], latent_2d[mask, 1], 
                   c=[palette[i]], label=f'Cluster {i}', alpha=0.7)
    ax.set_title(f't-SNE: {best_method} Predictions')
    ax.legend()
    
    ax = axes[1, 1]
    cm = confusion_matrix(genre_labels, pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=[f'C{i}' for i in range(n_genres)],
                yticklabels=genre_names)
    ax.set_title('Confusion Matrix')
    
    plt.tight_layout()
    plt.savefig(f'{results_dir}/full_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    model.eval()
    with torch.no_grad():
        sample = torch.FloatTensor(data[:4]).to(DEVICE)
        recon, _, _, _ = model(sample)
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 6))
        for i in range(4):
            axes[0, i].imshow(sample[i, 0].cpu().numpy(), aspect='auto', origin='lower')
            axes[0, i].set_title(f'Original {i+1}')
            axes[1, i].imshow(recon[i, 0].cpu().numpy(), aspect='auto', origin='lower')
            axes[1, i].set_title(f'Reconstructed {i+1}')
        plt.tight_layout()
        plt.savefig(f'{results_dir}/reconstructions.png', dpi=150)
        plt.close()


def main():
    print("=" * 60)
    print("MEDIUM TASK: CONV-VAE FOR MEL-SPECTROGRAM CLUSTERING")
    print("=" * 60)
    
    results_dir = CONFIG['results_dir']
    os.makedirs(results_dir, exist_ok=True)
    
    data, genre_labels, metadata = load_data(CONFIG['data_dir'])
    n_clusters = metadata['num_genres']
    
    data_tensor = torch.FloatTensor(data)
    dataset = TensorDataset(data_tensor)
    train_loader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True, drop_last=True)
    
    print("\nBuilding Conv-VAE...")
    model = ConvVAE(latent_dim=CONFIG['latent_dim']).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {total_params:,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2)
    
    print("\nTraining...")
    history = {'loss': [], 'recon': [], 'kl': []}
    best_loss = float('inf')
    
    for epoch in range(CONFIG['epochs']):
        loss, recon, kl = train_epoch(model, train_loader, optimizer, scheduler, CONFIG['beta'])
        
        history['loss'].append(loss)
        history['recon'].append(recon)
        history['kl'].append(kl)
        
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), f'{results_dir}/best_model.pth')
        
        if (epoch + 1) % 15 == 0:
            print(f"   Epoch {epoch+1}/{CONFIG['epochs']}: Loss={loss:.4f}")
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history['loss'], label='Total')
    ax.plot(history['recon'], label='Reconstruction')
    ax.plot(history['kl'], label='KL')
    ax.legend()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training History')
    plt.savefig(f'{results_dir}/training_history.png', dpi=150)
    plt.close()
    
    model.load_state_dict(torch.load(f'{results_dir}/best_model.pth'))
    
    print("\nExtracting latent representations...")
    latent = extract_latent(model, data_tensor)
    np.save(f'{results_dir}/latent_features.npy', latent)
    
    print(f"\nClustering with {n_clusters} clusters...")
    results, best_method = evaluate_clustering(latent, genre_labels, n_clusters)
    
    print("\n" + "=" * 60)
    print("CONV-VAE RESULTS")
    print("=" * 60)
    
    for method, metrics in results.items():
        marker = "*" if method == best_method else " "
        print(f"\n{marker} {method}:")
        print(f"   Silhouette: {metrics['silhouette']:.4f}")
        print(f"   ARI: {metrics['ari']:.4f}")
        print(f"   NMI: {metrics['nmi']:.4f}")
        print(f"   Purity: {metrics['purity']:.4f}")
    
    print("\nCreating visualizations...")
    visualize_results(latent, genre_labels, results, best_method, results_dir, metadata, data, model)
    
    final_results = {
        'config': CONFIG,
        'results': {
            method: {k: float(v) if isinstance(v, (np.floating, float)) else v 
                    for k, v in m.items() if k != 'predictions'}
            for method, m in results.items()
        },
        'best_method': best_method
    }
    
    with open(f'{results_dir}/results.json', 'w') as f:
        json.dump(final_results, f, indent=2)


if __name__ == '__main__':
    main()
