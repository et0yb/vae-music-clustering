import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from pathlib import Path
from typing import Optional, List
import torch

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("UMAP not available. Install with: pip install umap-learn")


def plot_tsne(features: np.ndarray,
              labels: np.ndarray,
              title: str = "t-SNE Visualization",
              label_names: Optional[List[str]] = None,
              save_path: Optional[str] = None,
              perplexity: int = 30,
              n_iter: int = 1000,
              figsize: tuple = (10, 8)):
    effective_perplexity = min(perplexity, len(features) - 1)
    tsne = TSNE(n_components=2, perplexity=effective_perplexity, max_iter=n_iter, random_state=42)
    features_2d = tsne.fit_transform(features)
    
    plt.figure(figsize=figsize)
    unique_labels = np.unique(labels)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        name = label_names[label] if label_names else f"Cluster {label}"
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                   c=[colors[i]], label=name, alpha=0.7, s=50)
    
    plt.title(title, fontsize=14)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    plt.close()


def plot_umap(features: np.ndarray,
              labels: np.ndarray,
              title: str = "UMAP Visualization",
              label_names: Optional[List[str]] = None,
              save_path: Optional[str] = None,
              n_neighbors: int = 15,
              min_dist: float = 0.1,
              figsize: tuple = (10, 8)):
    if not UMAP_AVAILABLE:
        print("UMAP not available. Using t-SNE instead.")
        return plot_tsne(features, labels, title, label_names, save_path)
    
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    features_2d = reducer.fit_transform(features)
    
    plt.figure(figsize=figsize)
    unique_labels = np.unique(labels)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        name = label_names[label] if label_names else f"Cluster {label}"
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                   c=[colors[i]], label=name, alpha=0.7, s=50)
    
    plt.title(title, fontsize=14)
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    plt.close()


def plot_cluster_comparison(features: np.ndarray,
                           true_labels: np.ndarray,
                           pred_labels: np.ndarray,
                           label_names: Optional[List[str]] = None,
                           save_path: Optional[str] = None,
                           figsize: tuple = (16, 6)):
    effective_perplexity = min(30, len(features) - 1)
    tsne = TSNE(n_components=2, perplexity=effective_perplexity, max_iter=1000, random_state=42)
    features_2d = tsne.fit_transform(features)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    ax = axes[0]
    unique_labels = np.unique(true_labels)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = true_labels == label
        name = label_names[label] if label_names else f"Class {label}"
        ax.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                  c=[colors[i]], label=name, alpha=0.7, s=50)
    
    ax.set_title("Ground Truth Labels", fontsize=12)
    ax.legend()
    
    ax = axes[1]
    unique_labels = np.unique(pred_labels)
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(unique_labels), 3)))
    
    for i, label in enumerate(unique_labels):
        mask = pred_labels == label
        name = "Noise" if label == -1 else f"Cluster {label}"
        ax.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                  c=[colors[i]], label=name, alpha=0.7, s=50)
    
    ax.set_title("Predicted Clusters", fontsize=12)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    plt.close()


def plot_reconstruction(original: np.ndarray,
                       reconstructed: np.ndarray,
                       n_samples: int = 5,
                       save_path: Optional[str] = None,
                       figsize: tuple = (15, 6)):
    if isinstance(original, torch.Tensor):
        original = original.cpu().numpy()
    if isinstance(reconstructed, torch.Tensor):
        reconstructed = reconstructed.cpu().numpy()
    
    if len(original.shape) == 4:
        original = original.squeeze(1)
        reconstructed = reconstructed.squeeze(1)
    
    n_samples = min(n_samples, len(original))
    fig, axes = plt.subplots(2, n_samples, figsize=figsize)
    
    for i in range(n_samples):
        axes[0, i].imshow(original[i], aspect='auto', origin='lower', cmap='viridis')
        axes[0, i].set_title(f"Original {i+1}")
        axes[0, i].axis('off')
        
        axes[1, i].imshow(reconstructed[i], aspect='auto', origin='lower', cmap='viridis')
        axes[1, i].set_title(f"Reconstructed {i+1}")
        axes[1, i].axis('off')
    
    axes[0, 0].set_ylabel("Original", fontsize=12)
    axes[1, 0].set_ylabel("Reconstructed", fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    plt.close()


def plot_latent_traversal(model, sample: torch.Tensor,
                          n_dims: int = 5,
                          range_val: float = 3.0,
                          steps: int = 10,
                          save_path: Optional[str] = None,
                          device: str = 'cuda'):
    model.eval()
    sample = sample.unsqueeze(0).to(device) if len(sample.shape) == 3 else sample.to(device)
    
    fig, axes = plt.subplots(n_dims, steps, figsize=(steps * 2, n_dims * 2))
    
    with torch.no_grad():
        mu, _ = model.encode(sample)
        
        for dim in range(n_dims):
            values = torch.linspace(-range_val, range_val, steps)
            
            for step_idx, val in enumerate(values):
                z = mu.clone()
                z[0, dim] = val
                recon = model.decode(z)
                
                img = recon[0].cpu().numpy()
                if len(img.shape) == 3:
                    img = img[0]
                
                axes[dim, step_idx].imshow(img, aspect='auto', origin='lower', cmap='viridis')
                axes[dim, step_idx].axis('off')
                
                if step_idx == 0:
                    axes[dim, step_idx].set_ylabel(f"Dim {dim}", fontsize=10)
    
    plt.suptitle("Latent Space Traversal", fontsize=14)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    plt.close()


def plot_metrics_comparison(metrics_df,
                           save_path: Optional[str] = None,
                           figsize: tuple = (12, 8)):
    import pandas as pd
    
    metric_cols = [c for c in metrics_df.columns if c not in ['method', 'error']]
    n_metrics = len(metric_cols)
    fig, axes = plt.subplots(2, (n_metrics + 1) // 2, figsize=figsize)
    axes = axes.flatten()
    
    for i, metric in enumerate(metric_cols):
        ax = axes[i]
        colors = plt.cm.Set2(np.linspace(0, 1, len(metrics_df)))
        bars = ax.bar(metrics_df['method'], metrics_df[metric], color=colors)
        
        ax.set_title(metric.replace('_', ' ').title())
        ax.set_ylabel("Score")
        ax.tick_params(axis='x', rotation=45)
        
        for bar, val in zip(bars, metrics_df[metric]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    for i in range(n_metrics, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    plt.close()


def plot_training_history(history: dict,
                         save_path: Optional[str] = None,
                         figsize: tuple = (12, 4)):
    n_plots = len(history)
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    
    if n_plots == 1:
        axes = [axes]
    
    for ax, (name, values) in zip(axes, history.items()):
        ax.plot(values)
        ax.set_title(name.replace('_', ' ').title())
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    plt.close()
