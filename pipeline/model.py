"""
3D U-Net Model Architecture and Checkpoint Management using MONAI and PyTorch.
Instantiates high-capacity volumetric 3D U-Nets with residual units and Instance Normalization.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn as nn
from monai.networks.nets import UNet

logger = logging.getLogger(__name__)


def build_3d_unet(
    in_channels: int = 1,
    out_channels: int = 2,
    channels: Sequence[int] = (16, 32, 64, 128, 256),
    strides: Sequence[int] = (2, 2, 2, 2),
    num_res_units: int = 2,
    norm: str = "instance",
    dropout: float = 0.1,
) -> UNet:
    """
    Instantiate a high-performance 3D U-Net from MONAI.

    Args:
        in_channels: Number of input volume channels (1 for single CT/MRI).
        out_channels: Number of segmentation output classes.
        channels: Feature channels per resolution level.
        strides: Downsampling strides per encoder level.
        num_res_units: Number of residual connection units per block.
        norm: Normalization layer ("instance" is standard for 3D medical images).
        dropout: Dropout probability.

    Returns:
        Instantiated monai.networks.nets.UNet.
    """
    logger.info(
        "Instantiating 3D U-Net: in_channels=%d, out_channels=%d, channels=%s, norm=%s",
        in_channels,
        out_channels,
        channels,
        norm,
    )
    model = UNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=channels,
        strides=strides,
        num_res_units=num_res_units,
        norm=norm,
        dropout=dropout,
    )
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("3D U-Net initialized with %d trainable parameters.", total_params)
    return model


def load_3d_unet(
    checkpoint_path: Optional[Path | str] = None,
    in_channels: int = 1,
    out_channels: int = 2,
    device: str = "cpu",
) -> nn.Module:
    """
    Build 3D U-Net and load weights from a checkpoint file if provided.
    If checkpoint_path is None or missing, returns initialized model for mock/testing.

    Args:
        checkpoint_path: Optional path to .pt / .pth model weights.
        in_channels: Number of input volume channels.
        out_channels: Number of output segmentation classes.
        device: Target execution device ('cuda', 'mps', 'cpu').

    Returns:
        Loaded PyTorch model in eval() mode on the specified device.
    """
    target_device = torch.device(device)
    model = build_3d_unet(in_channels=in_channels, out_channels=out_channels)

    if checkpoint_path is not None:
        ckpt_path = Path(checkpoint_path)
        if ckpt_path.is_file():
            logger.info("Loading model checkpoint from: '%s'", ckpt_path)
            state_dict = torch.load(ckpt_path, map_location=target_device)
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            model.load_state_dict(state_dict, strict=False)
            logger.info("Model weights loaded successfully.")
        else:
            logger.warning(
                "Checkpoint '%s' not found. Using freshly initialized 3D U-Net weights.",
                ckpt_path,
            )
    else:
        logger.info("No checkpoint specified. Operating with initialized 3D U-Net.")

    model.to(target_device)
    model.eval()
    return model
