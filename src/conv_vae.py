import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class ConvVAE(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 latent_dim: int = 64,
                 hidden_channels: List[int] = [32, 64, 128, 256],
                 input_size: Tuple[int, int] = (128, 431)):
        super(ConvVAE, self).__init__()
        
        self.latent_dim = latent_dim
        self.hidden_channels = hidden_channels
        self.input_size = input_size
        self.final_size = self._get_conv_output_size(input_size)
        
        encoder_layers = []
        in_ch = in_channels
        for h_ch in hidden_channels:
            encoder_layers.extend([
                nn.Conv2d(in_ch, h_ch, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(h_ch),
                nn.LeakyReLU(0.2)
            ])
            in_ch = h_ch
        
        self.encoder = nn.Sequential(*encoder_layers)
        self.flatten_size = hidden_channels[-1] * self.final_size[0] * self.final_size[1]
        
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_var = nn.Linear(self.flatten_size, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, self.flatten_size)
        
        decoder_layers = []
        hidden_channels_reversed = hidden_channels[::-1]
        
        for i in range(len(hidden_channels_reversed) - 1):
            decoder_layers.extend([
                nn.ConvTranspose2d(hidden_channels_reversed[i], 
                                   hidden_channels_reversed[i+1],
                                   kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(hidden_channels_reversed[i+1]),
                nn.LeakyReLU(0.2)
            ])
        
        decoder_layers.extend([
            nn.ConvTranspose2d(hidden_channels_reversed[-1], in_channels,
                              kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        ])
        
        self.decoder = nn.Sequential(*decoder_layers)
        
    def _get_conv_output_size(self, input_size: Tuple[int, int]) -> Tuple[int, int]:
        h, w = input_size
        for _ in self.hidden_channels:
            h = (h + 2 * 1 - 3) // 2 + 1
            w = (w + 2 * 1 - 3) // 2 + 1
        return (h, w)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        return mu, log_var
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder_input(z)
        h = h.view(h.size(0), self.hidden_channels[-1], self.final_size[0], self.final_size[1])
        return self.decoder(h)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z)
        
        if recon.size() != x.size():
            recon = F.interpolate(recon, size=x.size()[2:], mode='bilinear', align_corners=False)
        
        return recon, mu, log_var
    
    def loss_function(self, recon_x: torch.Tensor, x: torch.Tensor,
                     mu: torch.Tensor, log_var: torch.Tensor) -> dict:
        recon_loss = F.mse_loss(recon_x, x, reduction='sum') / x.size(0)
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / x.size(0)
        total_loss = recon_loss + kl_loss
        return {'loss': total_loss, 'recon_loss': recon_loss, 'kl_loss': kl_loss}
    
    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        mu, _ = self.encode(x)
        return mu


def train_conv_vae_epoch(model: ConvVAE, train_loader, optimizer, device: str = 'cuda') -> dict:
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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


def evaluate_conv_vae(model: ConvVAE, test_loader, device: str = 'cuda') -> dict:
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
