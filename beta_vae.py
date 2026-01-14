import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class BetaVAE(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 latent_dim: int = 64,
                 hidden_channels: List[int] = [32, 64, 128, 256],
                 beta: float = 4.0,
                 input_size: Tuple[int, int] = (128, 431)):
        super(BetaVAE, self).__init__()
        
        self.latent_dim = latent_dim
        self.hidden_channels = hidden_channels
        self.beta = beta
        self.input_size = input_size
        self.final_size = self._get_conv_output_size(input_size)
        
        self.encoder_blocks = nn.ModuleList()
        in_ch = in_channels
        for h_ch in hidden_channels:
            self.encoder_blocks.append(EncoderBlock(in_ch, h_ch))
            in_ch = h_ch
        
        self.flatten_size = hidden_channels[-1] * self.final_size[0] * self.final_size[1]
        
        self.fc_mu = nn.Sequential(
            nn.Linear(self.flatten_size, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim)
        )
        self.fc_var = nn.Sequential(
            nn.Linear(self.flatten_size, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim)
        )
        
        self.decoder_input = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, self.flatten_size)
        )
        
        self.decoder_blocks = nn.ModuleList()
        hidden_channels_reversed = hidden_channels[::-1]
        
        for i in range(len(hidden_channels_reversed) - 1):
            self.decoder_blocks.append(
                DecoderBlock(hidden_channels_reversed[i], hidden_channels_reversed[i+1])
            )
        
        self.final_decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels_reversed[-1], in_channels,
                              kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
        
    def _get_conv_output_size(self, input_size: Tuple[int, int]) -> Tuple[int, int]:
        h, w = input_size
        for _ in self.hidden_channels:
            h = (h + 2 * 1 - 3) // 2 + 1
            w = (w + 2 * 1 - 3) // 2 + 1
        return (h, w)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = x
        for block in self.encoder_blocks:
            h = block(h)
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
        for block in self.decoder_blocks:
            h = block(h)
        return self.final_decoder(h)
    
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
        total_loss = recon_loss + self.beta * kl_loss
        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'weighted_kl_loss': self.beta * kl_loss
        }
    
    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        mu, _ = self.encode(x)
        return mu
    
    def traverse_latent(self, x: torch.Tensor, dim: int, 
                       range_val: float = 3.0, steps: int = 10) -> torch.Tensor:
        mu, _ = self.encode(x)
        results = []
        values = torch.linspace(-range_val, range_val, steps)
        for val in values:
            z = mu.clone()
            z[:, dim] = val
            recon = self.decode(z)
            results.append(recon)
        return torch.stack(results, dim=1)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(EncoderBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
        
    def forward(self, x):
        return self.conv(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(DecoderBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
        
    def forward(self, x):
        return self.conv(x)


def train_beta_vae_epoch(model: BetaVAE, train_loader, optimizer, device: str = 'cuda') -> dict:
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


def evaluate_beta_vae(model: BetaVAE, test_loader, device: str = 'cuda') -> dict:
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
