"""
Denoising Diffusion (DDPM) from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - linear_beta_schedule
import torch
import torch.nn.functional as F

def linear_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02):
    return torch.linspace(beta_start, beta_end, T)

# Step 2 - alphas_from_betas
import torch
import torch.nn.functional as F

def alphas_from_betas(betas):
    return 1 - betas

# Step 3 - cumprod_alphas
import torch
import torch.nn.functional as F

def cumprod_alphas(alphas):
    return torch.cumprod(alphas, dim=0)

# Step 4 - extract_into_batch
import torch
import torch.nn.functional as F

def extract_into_batch(a, t, x):
    batch_size = t.shape[0]
    out = a.to(t.device).gather(0, t)

    return out.reshape(batch_size, *([1] * (x.dim() -1 )))

# Step 5 - q_sample
import torch
import torch.nn.functional as F

def q_sample(x0, t, noise, alphas_cumprod):
    sqrt_alpha_bar = extract_into_batch(alphas_cumprod.sqrt(), t, x0)
    sqrt_one_minus_alpha_bar = extract_into_batch((1 - alphas_cumprod).sqrt(), t, x0)

    return sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise

# Step 6 - build_diffusion_schedule
import torch
import torch.nn.functional as F

def build_diffusion_schedule(T: int = 100, beta_start: float = 1e-4, beta_end: float = 0.02) -> dict:
    betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float32)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    return {
        'betas': betas,
        'alphas': alphas,
        'alphas_cumprod': alphas_cumprod,
        'sqrt_alphas_cumprod': torch.sqrt(alphas_cumprod),
        'sqrt_one_minus_alphas_cumprod': torch.sqrt(1.0 - alphas_cumprod),
        'T': int(T),
    }

# Step 7 - noise_prediction_loss
import torch
import torch.nn.functional as F

def noise_prediction_loss(noise_pred, noise):
    return F.mse_loss(noise_pred, noise)

# Step 8 - diffusion_training_loss
import torch
import torch.nn.functional as F

def diffusion_training_loss(model, x0, t, noise, alphas_cumprod):
    x_t = q_sample(x0, t, noise, alphas_cumprod)
    noise_pred = model(x_t, t)

    return noise_prediction_loss(noise_pred, noise)

# Step 9 - timestep_embedding
import torch
import torch.nn.functional as F

def timestep_embedding(t, dim: int):
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device)
    )
    args = t.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    
    return emb

# Step 10 - init_tiny_unet
import torch
import torch.nn.functional as F

def init_tiny_unet(in_ch: int = 1, hidden: int = 16, time_dim: int = 16, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)

    def w(*shape):
        return (torch.randn(*shape, generator=g) * 0.02).requires_grad_()

    def b(n):
        return torch.zeros(n, requires_grad=True)

    return {
        'time_mlp_w': w(hidden, time_dim),
        'time_mlp_b': b(hidden),
        'conv_in_w':  w(hidden, in_ch, 3, 3),
        'conv_in_b':  b(hidden),
        'conv_mid_w': w(hidden, hidden, 3, 3),
        'conv_mid_b': b(hidden),
        'conv_out_w': w(in_ch, hidden, 3, 3),
        'conv_out_b': b(in_ch),
    }

# Step 11 - tiny_unet_forward
import torch
import torch.nn.functional as F

def tiny_unet_forward(x, t, params: dict):
    h = F.conv2d(x, params['conv_in_w'], params['conv_in_b'], padding=1)

    temb = timestep_embedding(t, params['time_mlp_w'].shape[1])
    temb = F.relu(F.linear(temb, params['time_mlp_w'], params['time_mlp_b']))

    h = h + temb[:, :, None, None]
    h = F.relu(h)
    h = F.relu(F.conv2d(h, params['conv_mid_w'], params['conv_mid_b'], padding=1))

    return F.conv2d(h, params['conv_out_w'], params['conv_out_b'], padding=1)

# Step 12 - make_blob_dataset (not yet solved)
# TODO: implement

# Step 13 - ddpm_train_step (not yet solved)
# TODO: implement

# Step 14 - train_ddpm (not yet solved)
# TODO: implement

# Step 15 - predict_x0_from_eps (not yet solved)
# TODO: implement

# Step 16 - ddpm_p_mean_variance (not yet solved)
# TODO: implement

# Step 17 - ddpm_p_sample (not yet solved)
# TODO: implement

# Step 18 - ddpm_sample_loop (not yet solved)
# TODO: implement

# Step 19 - sample_quality_mse (not yet solved)
# TODO: implement

# Step 20 - ddpm_experiment (not yet solved)
# TODO: implement

