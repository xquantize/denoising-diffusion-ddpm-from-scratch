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

# Step 7 - noise_prediction_loss (not yet solved)
# TODO: implement

# Step 8 - diffusion_training_loss (not yet solved)
# TODO: implement

# Step 9 - timestep_embedding (not yet solved)
# TODO: implement

# Step 10 - init_tiny_unet (not yet solved)
# TODO: implement

# Step 11 - tiny_unet_forward (not yet solved)
# TODO: implement

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

