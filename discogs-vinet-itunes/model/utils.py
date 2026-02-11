"""Simplified inference-only model loader for Discogs-VINet."""

import yaml
import torch

from model.nets import CQTNet


def build_model(config: dict, device: torch.device) -> CQTNet:
    model = CQTNet(
        ch_in=config["MODEL"]["CONV_CHANNEL"],
        ch_out=config["MODEL"]["EMBEDDING_SIZE"],
        norm=config["MODEL"]["NORMALIZATION"],
        pool=config["MODEL"]["POOLING"],
        l2_normalize=config["MODEL"]["L2_NORMALIZE"],
        projection=config["MODEL"]["PROJECTION"],
    )
    model.to(device)
    return model


def load_model(config: dict, device: torch.device) -> CQTNet:
    model = build_model(config, device)
    checkpoint_path = config["MODEL"]["CHECKPOINT_PATH"]
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model
