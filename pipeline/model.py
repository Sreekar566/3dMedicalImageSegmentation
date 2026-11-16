"""
3D U-Net Model Architecture and Checkpoint Management using MONAI and PyTorch.
Instantiates high-capacity volumetric 3D U-Nets with residual units and Instance Normalization.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn as nn
from monai.networks.nets import UNet

logger = logging.getLogger(__name__)

__all__ = [
    "build_3d_unet",
    "extract_state_dict",
    "load_3d_unet",
]


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
    Instantiate a high-performance 3D U-Net from MONAI with validated architecture parameters.

    Args:
        in_channels: Number of input volume channels (1 for single CT/MRI).
        out_channels: Number of segmentation output classes.
        channels: Feature channels per resolution level (at least 2 levels).
        strides: Downsampling strides per encoder level (length must equal len(channels) - 1).
        num_res_units: Number of residual connection units per block.
        norm: Normalization layer ("instance" is standard for 3D medical images).
        dropout: Dropout probability (between 0.0 and 1.0).

    Returns:
        Instantiated monai.networks.nets.UNet.
    """
    if not isinstance(in_channels, int) or in_channels <= 0:
        raise ValueError(f"in_channels must be a strictly positive integer, got {in_channels}.")
    if not isinstance(out_channels, int) or out_channels <= 0:
        raise ValueError(f"out_channels must be a strictly positive integer, got {out_channels}.")
    if not isinstance(num_res_units, int) or num_res_units < 0:
        raise ValueError(f"num_res_units must be a non-negative integer, got {num_res_units}.")
    if not isinstance(dropout, (int, float)) or not (0.0 <= float(dropout) <= 1.0):
        raise ValueError(f"dropout must be a float between 0.0 and 1.0, got {dropout}.")
    if not isinstance(channels, (list, tuple, Sequence)) or len(channels) < 2:
        raise ValueError(f"channels must contain at least 2 resolution levels, got {channels}.")
    if any(not isinstance(c, int) or c <= 0 for c in channels):
        raise ValueError(f"All channel dimensions must be strictly positive integers, got {channels}.")
    if not isinstance(strides, (list, tuple, Sequence)) or len(strides) != len(channels) - 1:
        raise ValueError(
            f"strides length ({len(strides)}) must equal len(channels) - 1 ({len(channels) - 1})."
        )
    if any(not isinstance(s, int) or s <= 0 for s in strides):
        raise ValueError(f"All stride values must be strictly positive integers, got {strides}.")
    if not isinstance(norm, str) or not norm.strip():
        raise ValueError(f"norm must be a non-empty string, got {norm!r}.")

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


def extract_state_dict(
    checkpoint: Any,
    target_keys: Sequence[str],
) -> Dict[str, Any]:
    """
    Extract and standardize a state dictionary from arbitrary PyTorch checkpoint formats.

    Handles:
    - Raw state_dict mapping (parameter names to Tensors).
    - Wrapped dictionary payloads with keys like 'state_dict', 'model_state_dict',
      'model', 'network', 'net', or 'weights'.
    - Serialized full nn.Module instances.
    - Stripping distributed / parallel training prefixes ('module.', '_orig_mod.', 'network.', 'net.').
    - Standardizing nested or missing 'model.' namespace prefixes.

    Args:
        checkpoint: Raw unpickled checkpoint payload.
        target_keys: Sequence of parameter names expected by the destination model.

    Returns:
        Cleaned state dictionary ready for model.load_state_dict().
    """
    if isinstance(checkpoint, nn.Module):
        checkpoint = checkpoint.state_dict()

    if not isinstance(checkpoint, dict):
        raise ValueError(
            f"Unexpected checkpoint payload type: {type(checkpoint).__name__}. "
            "Expected a dict or nn.Module."
        )

    # Unwrap nested state dict containers if present
    for wrapper in ("state_dict", "model_state_dict", "model", "network", "net", "weights"):
        if wrapper in checkpoint and isinstance(checkpoint[wrapper], dict):
            sub = checkpoint[wrapper]
            if any(isinstance(v, torch.Tensor) for v in sub.values()):
                checkpoint = sub
                break

    target_set = set(target_keys)
    if set(checkpoint.keys()).issubset(target_set) and len(checkpoint) > 0:
        return checkpoint

    cleaned: Dict[str, Any] = {}
    for k, v in checkpoint.items():
        if not isinstance(v, torch.Tensor):
            continue
        new_k = k
        for prefix in ("module.", "_orig_mod.", "network.", "net.", "model.model."):
            if new_k.startswith(prefix):
                if prefix == "model.model.":
                    new_k = new_k[len("model."):]
                else:
                    new_k = new_k[len(prefix):]
        if new_k not in target_set and ("model." + new_k) in target_set:
            new_k = "model." + new_k
        cleaned[new_k] = v

    return cleaned


def load_3d_unet(
    checkpoint_path: Optional[Path | str] = None,
    in_channels: int = 1,
    out_channels: int = 2,
    device: str | torch.device = "cpu",
    strict: bool = False,
) -> nn.Module:
    """
    Build 3D U-Net and load weights from a checkpoint file if provided.
    If checkpoint_path is None or missing, returns initialized model for mock/testing.

    Args:
        checkpoint_path: Optional path to .pt / .pth model weights.
        in_channels: Number of input volume channels.
        out_channels: Number of output segmentation classes.
        device: Target execution device ('cuda', 'mps', 'cpu' or torch.device).
        strict: Whether to require all keys in state_dict to strictly match model parameters.

    Returns:
        Loaded PyTorch model in eval() mode on the specified device.
    """
    if not isinstance(device, (str, torch.device)):
        raise TypeError(f"device must be a string or torch.device, got {type(device).__name__}.")
    try:
        target_device = torch.device(device)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"Invalid PyTorch execution device '{device}': {exc}") from exc

    resolved_ckpt: Optional[Path] = None
    if checkpoint_path is not None:
        if isinstance(checkpoint_path, str):
            clean = checkpoint_path.strip()
            if clean and clean.lower() not in ("none", "null"):
                resolved_ckpt = Path(clean).resolve()
        elif isinstance(checkpoint_path, Path):
            if str(checkpoint_path) not in ("", "."):
                resolved_ckpt = checkpoint_path.resolve()
        else:
            raise TypeError(
                f"checkpoint_path must be a Path, str, or None, got {type(checkpoint_path).__name__}."
            )

    model = build_3d_unet(in_channels=in_channels, out_channels=out_channels)

    if resolved_ckpt is not None:
        if resolved_ckpt.is_file():
            logger.info("Loading model checkpoint from: '%s'", resolved_ckpt)
            try:
                raw_payload = torch.load(resolved_ckpt, map_location=target_device, weights_only=True)
            except Exception:
                raw_payload = torch.load(resolved_ckpt, map_location=target_device, weights_only=False)

            target_keys = list(model.state_dict().keys())
            state_dict = extract_state_dict(raw_payload, target_keys=target_keys)

            matched_keys = set(state_dict.keys()) & set(target_keys)
            if not matched_keys:
                raise ValueError(
                    f"Failed to load checkpoint '{resolved_ckpt}': no matching weights found "
                    f"for 3D U-Net (target model expects {len(target_keys)} parameters)."
                )

            incompatible = model.load_state_dict(state_dict, strict=strict)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                logger.warning(
                    "Checkpoint partially matched: %d matched, %d missing keys, %d unexpected keys.",
                    len(matched_keys),
                    len(incompatible.missing_keys),
                    len(incompatible.unexpected_keys),
                )
            else:
                logger.info("Model weights loaded successfully (%d keys matched).", len(matched_keys))
        else:
            logger.warning(
                "Checkpoint '%s' not found. Using freshly initialized 3D U-Net weights.",
                resolved_ckpt,
            )
    else:
        logger.info("No checkpoint specified. Operating with initialized 3D U-Net.")

    model.to(target_device)
    model.eval()
    return model
