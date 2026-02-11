#!/usr/bin/env python3
"""
Utilities for loading CoverHunter model for inference.
"""

import os
import logging

import torch
import yaml

from model.model import Model


def load_model(checkpoint_dir, device='cpu'):
    """
    Load CoverHunter model from checkpoint directory.

    Args:
        checkpoint_dir: Directory containing hparams.yaml and g_* checkpoint file
        device: Device to load model on ('cpu', 'cuda', 'mps')

    Returns:
        model: Loaded CoverHunter model in eval mode
    """
    # Load hyperparameters
    hparams_path = os.path.join(checkpoint_dir, 'hparams.yaml')
    with open(hparams_path, 'r') as f:
        hp = yaml.safe_load(f)

    # Override device
    hp['device'] = device

    # Create model
    logging.info(f"Creating CoverHunter model with embed_dim={hp['embed_dim']}")
    model = Model(hp)

    # Load checkpoint
    model.load_model_parameters(checkpoint_dir, epoch_num=-1, device=device)
    model.to(device)
    model.eval()

    return model


def compute_embedding_from_cqt(model, cqt_features, device='cpu'):
    """
    Compute CoverHunter embedding from CQT features.

    Args:
        model: Loaded CoverHunter model
        cqt_features: CQT spectrogram as numpy array (time, freq) = (T, 96)
        device: Device to compute on

    Returns:
        embedding: 128-dim L2-normalized embedding as numpy array
    """
    import numpy as np

    # Prepare input tensor: (batch=1, time, freq)
    feat = torch.from_numpy(cqt_features).float().unsqueeze(0)
    feat = feat.to(device)

    # Run inference
    with torch.no_grad():
        embed, _ = model.inference(feat)

    # Convert to numpy and L2 normalize
    embed = embed.cpu().numpy().flatten()
    embed = embed / (np.linalg.norm(embed) + 1e-8)

    return embed
