import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class VAE(nn.Module):
    def __init__(self, 
                 input_dim: int,
                 hidden_dims: List[int] = [512, 256, 128],
                 latent_dim: int = 32):
        super(VAE, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        
        encoder_layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            in_dim = h_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)
        
        decoder_layers = []
        hidden_dims_reversed = hidden_dims[::-1]
        self.decoder_input = nn.Linear(latent_dim, hidden_dims_reversed[0])
        
        in_dim = hidden_dims_reversed[0]
        for h_dim in hidden_dims_reversed[1:]:
            decoder_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            in_dim = h_dim
        
        self.decoder = nn.Sequential(*decoder_layers)
        self.final_layer = nn.Linear(hidden_dims_reversed[-1], input_dim)
        
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        return mu, log_var
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder_input(z)
        h = F.relu(h)
        h = self.decoder(h)
        return self.final_layer(h)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z)
        return recon, mu, log_var
    
    def loss_function(self, recon_x: torch.Tensor, x: torch.Tensor,
                     mu: torch.Tensor, log_var: torch.Tensor) -> dict:
        recon_loss = F.mse_loss(recon_x, x, reduction='sum') / x.size(0)
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / x.size(0)
        total_loss = recon_loss + kl_loss
        return {'loss': total_loss, 'recon_loss': recon_loss, 'kl_loss': kl_loss}
    
    def sample(self, num_samples: int, device: str = 'cuda') -> torch.Tensor:
        z = torch.randn(num_samples, self.latent_dim).to(device)
        return self.decode(z)
    
    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        mu, _ = self.encode(x)
        return mu


def train_vae_epoch(model: VAE, train_loader, optimizer, device: str = 'cuda') -> dict:
    model.train()
    total_loss = 0
    total_recon = 0
    total_kl = 0
    
    for batch_data, _ in train_loader:
        batch_data = batch_data.to(device)
        optimizer.zero_grad()
        recon, mu, log_var = model(batch_data)
        losses = model.loss_function(recon, batch_data, mu, log_var)
        losses['loss'].backward()
        optimizer.step()
        
        total_loss += losses['loss'].item()
        total_recon += losses['recon_loss'].item()
        total_kl += losses['kl_loss'].item()
    
    n_batches = len(train_loader)
    return {
        'loss': total_loss / n_batches,
        'recon_loss': total_recon / n_batches,
        'kl_loss': total_kl / n_batches
    }


def evaluate_vae(model: VAE, test_loader, device: str = 'cuda') -> dict:
    model.eval()
    total_loss = 0
    total_recon = 0
    total_kl = 0
    
    with torch.no_grad():
        for batch_data, _ in test_loader:
            batch_data = batch_data.to(device)
            recon, mu, log_var = model(batch_data)
            losses = model.loss_function(recon, batch_data, mu, log_var)
            
            total_loss += losses['loss'].item()
            total_recon += losses['recon_loss'].item()
            total_kl += losses['kl_loss'].item()
    
    n_batches = len(test_loader)
    return {
        'loss': total_loss / n_batches,
        'recon_loss': total_recon / n_batches,
        'kl_loss': total_kl / n_batches
    }
