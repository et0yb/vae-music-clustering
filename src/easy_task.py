import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    adjusted_rand_score, normalized_mutual_info_score, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
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
    'input_dim': None,
    'hidden_dims': [1024, 512, 256],
    'latent_dim': 64,
    'epochs': 200,
    'batch_size': 64,
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
    'beta_start': 0.0,
    'beta_end': 1.0,
    'beta_warmup_epochs': 50,
    'n_clusters': 7,
    'data_dir': 'data/processed',
    'results_dir': 'results/easy'
}


class ImprovedVAE(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, dropout=0.2):
        super().__init__()
        
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout)
            ])
            prev_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)
        
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        decoder_layers = []
        hidden_dims_rev = hidden_dims[::-1]
        prev_dim = latent_dim
        for h_dim in hidden_dims_rev:
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout)
            ])
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(hidden_dims_rev[-1], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z


def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


def load_and_preprocess_data(data_dir):
    print("\nLoading data...")
    
    combined = np.load(f'{data_dir}/combined_features.npy')
    genre_labels = np.load(f'{data_dir}/genre_labels.npy')
    
    with open(f'{data_dir}/metadata.json', 'r') as f:
        metadata = json.load(f)
    
    print(f"   Combined features shape: {combined.shape}")
    print(f"   Genres: {metadata['num_genres']}")
    print(f"   Total samples: {len(genre_labels)}")
    
    scaler = StandardScaler()
    combined_scaled = scaler.fit_transform(combined)
    
    return combined_scaled, genre_labels, metadata, scaler


def train_vae(model, train_loader, optimizer, scheduler, epoch, config):
    model.train()
    total_loss = 0
    total_recon = 0
    total_kl = 0
    
    if epoch < config['beta_warmup_epochs']:
        beta = config['beta_start'] + (config['beta_end'] - config['beta_start']) * (epoch / config['beta_warmup_epochs'])
    else:
        beta = config['beta_end']
    
    for batch_idx, (data,) in enumerate(train_loader):
        data = data.to(DEVICE)
        optimizer.zero_grad()
        
        recon, mu, logvar, z = model(data)
        loss, recon_loss, kl_loss = vae_loss(recon, data, mu, logvar, beta)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()
    
    scheduler.step()
    
    n = len(train_loader.dataset)
    return total_loss / n, total_recon / n, total_kl / n, beta


def extract_latent(model, data_tensor):
    model.eval()
    with torch.no_grad():
        data = data_tensor.to(DEVICE)
        mu, _ = model.encode(data)
        return mu.cpu().numpy()


def cluster_and_evaluate(latent, true_labels, n_clusters, label_name=""):
    results = {}
    
    clustering_methods = {
        'K-Means': KMeans(n_clusters=n_clusters, random_state=42, n_init=20),
        'GMM': GaussianMixture(n_components=n_clusters, random_state=42, n_init=5),
        'Agglomerative': AgglomerativeClustering(n_clusters=n_clusters, linkage='ward'),
        'Spectral': SpectralClustering(n_clusters=n_clusters, random_state=42, affinity='nearest_neighbors')
    }
    
    best_method = None
    best_ari = -1
    
    for name, clusterer in clustering_methods.items():
        try:
            if name == 'GMM':
                pred_labels = clusterer.fit_predict(latent)
            else:
                pred_labels = clusterer.fit_predict(latent)
            
            sil = silhouette_score(latent, pred_labels)
            ch = calinski_harabasz_score(latent, pred_labels)
            db = davies_bouldin_score(latent, pred_labels)
            ari = adjusted_rand_score(true_labels, pred_labels)
            nmi = normalized_mutual_info_score(true_labels, pred_labels)
            purity = compute_purity(true_labels, pred_labels)
            
            results[name] = {
                'silhouette': sil,
                'calinski_harabasz': ch,
                'davies_bouldin': db,
                'adjusted_rand_index': ari,
                'normalized_mutual_info': nmi,
                'purity': purity,
                'predictions': pred_labels
            }
            
            if ari > best_ari:
                best_ari = ari
                best_method = name
                
        except Exception as e:
            print(f"   Warning: {name} failed - {e}")
    
    return results, best_method


def compute_purity(true_labels, pred_labels):
    contingency = confusion_matrix(true_labels, pred_labels)
    return np.sum(np.max(contingency, axis=0)) / len(true_labels)


def visualize_results(latent, genre_labels, clustering_results, best_method, results_dir, metadata):
    os.makedirs(results_dir, exist_ok=True)
    
    n_genres = metadata['num_genres']
    idx_to_genre = metadata['idx_to_genre']
    genre_names = [idx_to_genre[str(i)] for i in range(n_genres)]
    
    print("   Computing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    latent_2d_tsne = tsne.fit_transform(latent)
    
    print("   Computing UMAP...")
    reducer = umap.UMAP(n_components=2, random_state=42)
    latent_2d_umap = reducer.fit_transform(latent)
    
    genre_palette = sns.color_palette("husl", n_genres)
    cluster_palette = sns.color_palette("Set2", n_genres)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    ax = axes[0, 0]
    for i in range(n_genres):
        mask = genre_labels == i
        ax.scatter(latent_2d_tsne[mask, 0], latent_2d_tsne[mask, 1], 
                   c=[genre_palette[i]], label=genre_names[i], alpha=0.7, s=50)
    ax.set_title('t-SNE: Ground Truth Genres', fontsize=14)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    
    ax = axes[0, 1]
    for i in range(n_genres):
        mask = genre_labels == i
        ax.scatter(latent_2d_umap[mask, 0], latent_2d_umap[mask, 1],
                   c=[genre_palette[i]], label=genre_names[i], alpha=0.7, s=50)
    ax.set_title('UMAP: Ground Truth Genres', fontsize=14)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    
    ax = axes[1, 0]
    pred = clustering_results[best_method]['predictions']
    for i in range(n_genres):
        mask = pred == i
        ax.scatter(latent_2d_tsne[mask, 0], latent_2d_tsne[mask, 1],
                   c=[cluster_palette[i]], label=f'Cluster {i}', alpha=0.7, s=50)
    ax.set_title(f't-SNE: {best_method} Predictions', fontsize=14)
    ax.legend()
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    
    ax = axes[1, 1]
    for i in range(n_genres):
        mask = pred == i
        ax.scatter(latent_2d_umap[mask, 0], latent_2d_umap[mask, 1],
                   c=[cluster_palette[i]], label=f'Cluster {i}', alpha=0.7, s=50)
    ax.set_title(f'UMAP: {best_method} Predictions', fontsize=14)
    ax.legend()
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    
    plt.tight_layout()
    plt.savefig(f'{results_dir}/latent_space_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    cm = confusion_matrix(genre_labels, pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=[f'C{i}' for i in range(n_genres)],
                yticklabels=genre_names)
    ax.set_title(f'Confusion Matrix: {best_method}', fontsize=14)
    ax.set_xlabel('Predicted Cluster')
    ax.set_ylabel('True Genre')
    plt.tight_layout()
    plt.savefig(f'{results_dir}/confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    metrics = ['silhouette', 'calinski_harabasz', 'adjusted_rand_index', 
               'normalized_mutual_info', 'purity', 'davies_bouldin']
    titles = ['Silhouette Score', 'Calinski-Harabasz', 'Adjusted Rand Index',
              'Normalized Mutual Info', 'Cluster Purity', 'Davies-Bouldin']
    
    methods = list(clustering_results.keys())
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 3, idx % 3]
        values = [clustering_results[m][metric] for m in methods]
        colors = ['#2ecc71' if m == best_method else '#3498db' for m in methods]
        bars = ax.bar(methods, values, color=colors)
        ax.set_title(title, fontsize=12)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Clustering Metrics Comparison', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{results_dir}/metrics_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()


def main():
    print("=" * 60)
    print("EASY TASK: VAE FOR FMA MUSIC CLUSTERING")
    print("=" * 60)
    
    results_dir = CONFIG['results_dir']
    os.makedirs(results_dir, exist_ok=True)
    
    data, genre_labels, metadata, scaler = load_and_preprocess_data(CONFIG['data_dir'])
    
    n_clusters = metadata['num_genres']
    CONFIG['n_clusters'] = n_clusters
    CONFIG['input_dim'] = data.shape[1]
    
    data_tensor = torch.FloatTensor(data)
    dataset = TensorDataset(data_tensor)
    train_loader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    
    print("\nBuilding model...")
    model = ImprovedVAE(
        input_dim=CONFIG['input_dim'],
        hidden_dims=CONFIG['hidden_dims'],
        latent_dim=CONFIG['latent_dim'],
        dropout=0.2
    ).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
    
    print("\nTraining...")
    history = {'loss': [], 'recon': [], 'kl': [], 'beta': []}
    best_loss = float('inf')
    
    for epoch in range(CONFIG['epochs']):
        loss, recon, kl, beta = train_vae(model, train_loader, optimizer, scheduler, epoch, CONFIG)
        
        history['loss'].append(loss)
        history['recon'].append(recon)
        history['kl'].append(kl)
        history['beta'].append(beta)
        
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), f'{results_dir}/best_model.pth')
        
        if (epoch + 1) % 20 == 0:
            print(f"   Epoch {epoch+1:3d}/{CONFIG['epochs']}: Loss={loss:.4f}, Recon={recon:.4f}, KL={kl:.4f}")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(history['loss'], label='Total Loss')
    axes[0].set_title('Total Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    
    axes[1].plot(history['recon'], label='Reconstruction', color='blue')
    axes[1].plot(history['kl'], label='KL Divergence', color='red')
    axes[1].set_title('Loss Components')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    
    axes[2].plot(history['beta'], label='Beta', color='green')
    axes[2].set_title('KL Annealing (Beta)')
    axes[2].set_xlabel('Epoch')
    
    plt.tight_layout()
    plt.savefig(f'{results_dir}/training_history.png', dpi=150)
    plt.close()
    
    model.load_state_dict(torch.load(f'{results_dir}/best_model.pth'))
    
    print("\nExtracting latent representations...")
    latent = extract_latent(model, data_tensor)
    np.save(f'{results_dir}/latent_features.npy', latent)
    print(f"   Latent shape: {latent.shape}")
    
    print(f"\nClustering with {n_clusters} clusters...")
    results, best_method = cluster_and_evaluate(latent, genre_labels, n_clusters)
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    for method, metrics in results.items():
        marker = "*" if method == best_method else " "
        print(f"\n{marker} {method}:")
        print(f"   Silhouette Score:      {metrics['silhouette']:.4f}")
        print(f"   Calinski-Harabasz:     {metrics['calinski_harabasz']:.2f}")
        print(f"   Davies-Bouldin:        {metrics['davies_bouldin']:.4f}")
        print(f"   Adjusted Rand Index:   {metrics['adjusted_rand_index']:.4f}")
        print(f"   Normalized MI:         {metrics['normalized_mutual_info']:.4f}")
        print(f"   Cluster Purity:        {metrics['purity']:.4f} ({metrics['purity']*100:.1f}%)")
    
    print("\nCreating visualizations...")
    visualize_results(latent, genre_labels, results, best_method, results_dir, metadata)
    
    final_results = {
        'config': CONFIG,
        'results': {
            method: {k: float(v) if isinstance(v, (np.floating, float)) else v 
                    for k, v in metrics.items() if k != 'predictions'}
            for method, metrics in results.items()
        },
        'best_method': best_method,
        'training_history': {k: [float(x) for x in v] for k, v in history.items()}
    }
    
    with open(f'{results_dir}/results.json', 'w') as f:
        json.dump(final_results, f, indent=2)


if __name__ == '__main__':
    main()
