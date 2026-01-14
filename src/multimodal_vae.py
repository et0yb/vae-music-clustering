import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List


class MultiModalVAE(nn.Module):
    def __init__(self,
                 audio_input_dim: int,
                 lyrics_input_dim: int,
                 audio_hidden_dims: List[int] = [512, 256],
                 lyrics_hidden_dims: List[int] = [128, 64],
                 latent_dim: int = 32,
                 fusion_method: str = 'concat'):
        super().__init__()
        
        self.audio_input_dim = audio_input_dim
        self.lyrics_input_dim = lyrics_input_dim
        self.latent_dim = latent_dim
        self.fusion_method = fusion_method
        
        audio_layers = []
        prev_dim = audio_input_dim
        for h_dim in audio_hidden_dims:
            audio_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.2)
            ])
            prev_dim = h_dim
        self.audio_encoder = nn.Sequential(*audio_layers)
        self.audio_latent_dim = audio_hidden_dims[-1]
        
        lyrics_layers = []
        prev_dim = lyrics_input_dim
        for h_dim in lyrics_hidden_dims:
            lyrics_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.2)
            ])
            prev_dim = h_dim
        self.lyrics_encoder = nn.Sequential(*lyrics_layers)
        self.lyrics_latent_dim = lyrics_hidden_dims[-1]
        
        if fusion_method == 'concat':
            fusion_dim = self.audio_latent_dim + self.lyrics_latent_dim
        else:
            fusion_dim = self.audio_latent_dim
        
        self.fc_mu = nn.Linear(fusion_dim, latent_dim)
        self.fc_var = nn.Linear(fusion_dim, latent_dim)
        
        audio_decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(audio_hidden_dims):
            audio_decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.2)
            ])
            prev_dim = h_dim
        audio_decoder_layers.append(nn.Linear(prev_dim, audio_input_dim))
        self.audio_decoder = nn.Sequential(*audio_decoder_layers)
        
        lyrics_decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(lyrics_hidden_dims):
            lyrics_decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.2)
            ])
            prev_dim = h_dim
        lyrics_decoder_layers.append(nn.Linear(prev_dim, lyrics_input_dim))
        self.lyrics_decoder = nn.Sequential(*lyrics_decoder_layers)
        
    def encode(self, audio: torch.Tensor, lyrics: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        audio_h = self.audio_encoder(audio)
        lyrics_h = self.lyrics_encoder(lyrics)
        
        if self.fusion_method == 'concat':
            fused = torch.cat([audio_h, lyrics_h], dim=1)
        elif self.fusion_method == 'add':
            fused = audio_h + lyrics_h
        else:
            fused = audio_h * lyrics_h
        
        mu = self.fc_mu(fused)
        log_var = self.fc_var(fused)
        return mu, log_var
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        audio_recon = self.audio_decoder(z)
        lyrics_recon = self.lyrics_decoder(z)
        return audio_recon, lyrics_recon
    
    def forward(self, audio: torch.Tensor, lyrics: torch.Tensor) -> dict:
        mu, log_var = self.encode(audio, lyrics)
        z = self.reparameterize(mu, log_var)
        audio_recon, lyrics_recon = self.decode(z)
        return {
            'audio_recon': audio_recon,
            'lyrics_recon': lyrics_recon,
            'mu': mu,
            'log_var': log_var,
            'z': z
        }
    
    def get_latent(self, audio: torch.Tensor, lyrics: torch.Tensor) -> torch.Tensor:
        mu, _ = self.encode(audio, lyrics)
        return mu


class AudioOnlyVAE(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int] = [512, 256, 128],
                 latent_dim: int = 32):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.2)
            ])
            prev_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)
        
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)
        
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.2)
            ])
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> dict:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z)
        return {'recon': recon, 'mu': mu, 'log_var': log_var, 'z': z}
    
    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        mu, _ = self.encode(x)
        return mu


class StandardAutoencoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int] = [512, 256, 128],
                 latent_dim: int = 32):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU()
            ])
            prev_dim = h_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU()
            ])
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> dict:
        z = self.encode(x)
        recon = self.decode(z)
        return {'recon': recon, 'z': z}
    
    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)


def multimodal_vae_loss(outputs: dict, 
                        audio_target: torch.Tensor,
                        lyrics_target: torch.Tensor,
                        audio_weight: float = 1.0,
                        lyrics_weight: float = 0.5,
                        kl_weight: float = 1.0) -> Tuple[torch.Tensor, dict]:
    audio_recon_loss = F.mse_loss(outputs['audio_recon'], audio_target, reduction='mean')
    lyrics_recon_loss = F.mse_loss(outputs['lyrics_recon'], lyrics_target, reduction='mean')
    
    mu = outputs['mu']
    log_var = outputs['log_var']
    kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    
    total_loss = (audio_weight * audio_recon_loss + 
                  lyrics_weight * lyrics_recon_loss + 
                  kl_weight * kl_loss)
    
    loss_dict = {
        'total': total_loss.item(),
        'audio_recon': audio_recon_loss.item(),
        'lyrics_recon': lyrics_recon_loss.item(),
        'kl': kl_loss.item()
    }
    
    return total_loss, loss_dict


def train_multimodal_vae_epoch(model: MultiModalVAE,
                               train_loader,
                               optimizer: torch.optim.Optimizer,
                               device: torch.device,
                               audio_weight: float = 1.0,
                               lyrics_weight: float = 0.5,
                               kl_weight: float = 1.0) -> dict:
    model.train()
    epoch_losses = {'total': 0, 'audio_recon': 0, 'lyrics_recon': 0, 'kl': 0}
    n_batches = 0
    
    for batch in train_loader:
        audio = batch['audio'].to(device)
        lyrics = batch['lyrics'].to(device)
        
        optimizer.zero_grad()
        outputs = model(audio, lyrics)
        loss, loss_dict = multimodal_vae_loss(
            outputs, audio, lyrics,
            audio_weight, lyrics_weight, kl_weight
        )
        
        loss.backward()
        optimizer.step()
        
        for k, v in loss_dict.items():
            epoch_losses[k] += v
        n_batches += 1
    
    for k in epoch_losses:
        epoch_losses[k] /= n_batches
    
    return epoch_losses


def evaluate_multimodal_vae(model: MultiModalVAE,
                            test_loader,
                            device: torch.device) -> dict:
    model.eval()
    all_latents = []
    all_genres = []
    all_languages = []
    epoch_losses = {'total': 0, 'audio_recon': 0, 'lyrics_recon': 0, 'kl': 0}
    n_batches = 0
    
    with torch.no_grad():
        for batch in test_loader:
            audio = batch['audio'].to(device)
            lyrics = batch['lyrics'].to(device)
            
            outputs = model(audio, lyrics)
            loss, loss_dict = multimodal_vae_loss(outputs, audio, lyrics)
            
            for k, v in loss_dict.items():
                epoch_losses[k] += v
            n_batches += 1
            
            all_latents.append(outputs['z'].cpu().numpy())
            all_genres.append(batch['genre'].numpy())
            all_languages.append(batch['language'].numpy())
    
    for k in epoch_losses:
        epoch_losses[k] /= n_batches
    
    import numpy as np
    latents = np.concatenate(all_latents, axis=0)
    genres = np.concatenate(all_genres, axis=0)
    languages = np.concatenate(all_languages, axis=0)
    
    return {
        'losses': epoch_losses,
        'latents': latents,
        'genres': genres,
        'languages': languages
    }
