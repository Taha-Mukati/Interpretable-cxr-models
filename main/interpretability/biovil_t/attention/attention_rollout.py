from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


def _ensure_batched_attn(attn: torch.Tensor) -> torch.Tensor:
    """
    Accepts:
      [H, N, N] or [B, H, N, N]
    Returns:
      [B, H, N, N]
    """
    if attn.ndim == 3:
        return attn.unsqueeze(0)
    if attn.ndim == 4:
        return attn
    raise ValueError(f"Expected [H,N,N] or [B,H,N,N], got {tuple(attn.shape)}")


def fuse_heads(attn: torch.Tensor, method: str = "mean") -> torch.Tensor:
    """
    attn: [B, H, N, N]
    returns: [B, N, N]
    """
    attn = _ensure_batched_attn(attn)

    if method == "mean":
        return attn.mean(dim=1)
    if method == "max":
        return attn.max(dim=1).values
    if method == "min":
        return attn.min(dim=1).values

    raise ValueError(f"Unsupported head fusion method: {method}")


def normalize_attention(
    attn: torch.Tensor,
    add_identity: bool = True,
    clamp_min: float = 1e-8,
) -> torch.Tensor:
    """
    attn: [B, N, N]
    returns: row-normalized attention [B, N, N]
    """
    if attn.ndim != 3:
        raise ValueError(f"Expected [B,N,N], got {tuple(attn.shape)}")

    if add_identity:
        eye = torch.eye(attn.size(-1), device=attn.device, dtype=attn.dtype)
        attn = attn + eye.unsqueeze(0)

    attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(clamp_min)
    return attn


def attention_rollout(
    attentions: List[torch.Tensor],
    add_identity: bool = True,
    head_fusion: str = "mean",
) -> torch.Tensor:
    """
    Generic transformer attention rollout.

    attentions:
        list of layer attentions, each [H,N,N] or [B,H,N,N]

    returns:
        rollout: [B,N,N]
    """
    if not attentions:
        raise ValueError("attentions list is empty")

    fused_layers = []
    for attn in attentions:
        fused = fuse_heads(attn, method=head_fusion)   # [B,N,N]
        fused = normalize_attention(fused, add_identity=add_identity)
        fused_layers.append(fused)

    rollout = fused_layers[0]
    for layer_attn in fused_layers[1:]:
        rollout = layer_attn @ rollout

    return rollout


# -----------------------------
# Spatial rollout helpers
# -----------------------------

def spatial_rollout_no_cls(
    rollout: torch.Tensor,
    grid_size: int = 14,
    reduce: str = "mean",
) -> torch.Tensor:
    """
    For patch-only attention matrices with no CLS token.

    rollout: [B, P, P]
    returns: [B, grid_size, grid_size]
    """
    if rollout.ndim != 3:
        raise ValueError(f"Expected [B,P,P], got {tuple(rollout.shape)}")

    if reduce == "mean":
        vec = rollout.mean(dim=1)   # [B, P]
    elif reduce == "max":
        vec = rollout.max(dim=1).values
    else:
        raise ValueError(f"Unsupported reduce method: {reduce}")

    expected = grid_size * grid_size
    if vec.shape[-1] != expected:
        raise ValueError(
            f"Expected {expected} patch tokens for grid_size={grid_size}, "
            f"got {vec.shape[-1]}"
        )

    return vec.view(vec.size(0), grid_size, grid_size)


def upsample_heatmap(
    heatmap: torch.Tensor,
    size: Tuple[int, int] = (448, 448),
) -> torch.Tensor:
    """
    heatmap: [B,H,W] or [H,W]
    returns: upsampled heatmap
    """
    squeeze = False
    if heatmap.ndim == 2:
        heatmap = heatmap.unsqueeze(0)
        squeeze = True

    if heatmap.ndim != 3:
        raise ValueError(f"Expected [H,W] or [B,H,W], got {tuple(heatmap.shape)}")

    up = F.interpolate(
        heatmap.unsqueeze(1),
        size=size,
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)

    return up[0] if squeeze else up


# -----------------------------
# Temporal rollout helpers
# -----------------------------

def split_temporal_rollout(
    rollout: torch.Tensor,
    current_length: int = 196,
    previous_length: int = 196,
    current_first: bool = True,
    num_prefix_tokens: int = 0,
) -> Dict[str, torch.Tensor]:
    """
    Split temporal rollout into quadrants.

    For your BioViL-T run:
      current_length=196
      previous_length=196
      num_prefix_tokens=0
    """
    if rollout.ndim == 2:
        rollout = rollout.unsqueeze(0)

    if rollout.ndim != 3:
        raise ValueError(f"Expected [N,N] or [B,N,N], got {tuple(rollout.shape)}")

    total_expected = num_prefix_tokens + current_length + previous_length
    if rollout.shape[-1] != total_expected:
        raise ValueError(
            f"Rollout token count mismatch: expected {total_expected}, "
            f"got {rollout.shape[-1]}"
        )

    start = num_prefix_tokens

    if current_first:
        c0, c1 = start, start + current_length
        p0, p1 = c1, c1 + previous_length
    else:
        p0, p1 = start, start + previous_length
        c0, c1 = p1, p1 + current_length

    return {
        "current_to_current": rollout[:, c0:c1, c0:c1],
        "current_to_previous": rollout[:, c0:c1, p0:p1],
        "previous_to_current": rollout[:, p0:p1, c0:c1],
        "previous_to_previous": rollout[:, p0:p1, p0:p1],
    }


def row_to_patch_heatmap(
    mat: torch.Tensor,
    row_index: int,
    grid_size: int = 14,
) -> torch.Tensor:
    """
    Select one row from [L,L] or [1,L,L] and reshape into [14,14].
    """
    if mat.ndim == 3:
        mat = mat[0]

    if mat.ndim != 2:
        raise ValueError(f"Expected [L,L], got {tuple(mat.shape)}")

    if row_index < 0 or row_index >= mat.size(0):
        raise IndexError(f"row_index {row_index} out of range for shape {tuple(mat.shape)}")

    vec = mat[row_index]
    expected = grid_size * grid_size
    if vec.numel() != expected:
        raise ValueError(f"Cannot reshape {vec.numel()} values into {grid_size}x{grid_size}")

    return vec.view(grid_size, grid_size)


def mean_patch_heatmap(
    mat: torch.Tensor,
    grid_size: int = 14,
) -> torch.Tensor:
    """
    Average rows of [L,L] or [1,L,L] to get one overall heatmap [14,14].
    """
    if mat.ndim == 3:
        mat = mat[0]

    if mat.ndim != 2:
        raise ValueError(f"Expected [L,L], got {tuple(mat.shape)}")

    vec = mat.mean(dim=0)
    expected = grid_size * grid_size
    if vec.numel() != expected:
        raise ValueError(f"Cannot reshape {vec.numel()} values into {grid_size}x{grid_size}")

    return vec.view(grid_size, grid_size)