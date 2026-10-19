from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def _get_image_backbone_and_projection(model):
    """
    Get the real torchvision ResNet backbone from the repo wrapper.
    """
    image_encoder = model.image_encoder

    if hasattr(image_encoder, "resnet"):
        backbone = image_encoder.resnet
    elif hasattr(image_encoder, "backbone"):
        backbone = image_encoder.backbone
    elif hasattr(image_encoder, "model"):
        backbone = image_encoder.model
    elif hasattr(image_encoder, "encoder"):
        backbone = image_encoder.encoder
    else:
        backbone = image_encoder

    if not hasattr(model, "image_projection"):
        raise ValueError("Model does not expose image_projection")

    return backbone, model.image_projection


def _get_last_conv_features(backbone, image_tensor: torch.Tensor) -> torch.Tensor:
    """
    Forward through a torchvision ResNet backbone up to layer4.
    Returns feature map [B, C, H, W].
    """
    x = image_tensor

    x = backbone.conv1(x)
    x = backbone.bn1(x)
    x = backbone.relu(x)
    x = backbone.maxpool(x)

    x = backbone.layer1(x)
    x = backbone.layer2(x)
    x = backbone.layer3(x)
    x = backbone.layer4(x)

    return x


@torch.no_grad()
def _get_label_text_embeddings(
    pipeline: Dict,
    label: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      pos_emb_mean: [1, D]
      neg_emb_mean: [1, D]
    """
    text_cache = pipeline.get("text_embedding_cache", None)

    if text_cache is not None and label in text_cache:
        pos_emb = text_cache[label]["pos"]
        neg_emb = text_cache[label]["neg"]
    else:
        from pipelines.cxr_clip_pipeline import get_text_embeddings_for_label
        embs = get_text_embeddings_for_label(pipeline, label)
        pos_emb = embs["pos"]
        neg_emb = embs["neg"]

    pos_emb = F.normalize(pos_emb, dim=-1).mean(dim=0, keepdim=True)
    neg_emb = F.normalize(neg_emb, dim=-1).mean(dim=0, keepdim=True)

    return pos_emb, neg_emb


@torch.no_grad()
def generate_similarity_relevance_map(
    pipeline: Dict,
    image_tensor: torch.Tensor,
    label: str,
    use_margin: bool = True,
) -> np.ndarray:
    """
    Generate a similarity-based relevance map for one image and one label.

    Parameters
    ----------
    pipeline:
        Your initialized CXR-CLIP pipeline dict.
    image_tensor:
        Preprocessed tensor [3, H, W]
    label:
        Pathology label
    use_margin:
        If True, map = sim(pos) - sim(neg)
        If False, map = sim(pos)

    Returns
    -------
    heatmap:
        numpy array [H, W], normalized to [0, 1]
    """
    model = pipeline["model"]
    device = next(model.parameters()).device

    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)

    backbone, image_projection = _get_image_backbone_and_projection(model)
    feat = _get_last_conv_features(backbone, image_tensor)   # [1, C, Hf, Wf]

    b, c, hf, wf = feat.shape

    # Convert each spatial location into a projected embedding
    spatial = feat.permute(0, 2, 3, 1).reshape(-1, c)        # [Hf*Wf, C]
    spatial_proj = image_projection(spatial)                 # [Hf*Wf, D]
    spatial_proj = F.normalize(spatial_proj, dim=-1)

    pos_emb, neg_emb = _get_label_text_embeddings(pipeline, label)
    pos_emb = pos_emb.to(device)
    neg_emb = neg_emb.to(device)

    pos_scores = spatial_proj @ pos_emb.T                    # [Hf*Wf, 1]

    if use_margin:
        neg_scores = spatial_proj @ neg_emb.T                # [Hf*Wf, 1]
        scores = pos_scores - neg_scores
    else:
        scores = pos_scores

    scores = scores.reshape(1, 1, hf, wf)
    scores = F.interpolate(
        scores,
        size=image_tensor.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )

    heatmap = scores[0, 0].detach().cpu().numpy()

    # normalize to [0, 1]
    if heatmap.max() > heatmap.min():
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    else:
        heatmap = np.zeros_like(heatmap)

    return heatmap


def show_similarity_relevance_map(
    image_tensor: torch.Tensor,
    heatmap: np.ndarray,
    alpha: float = 0.4,
):
    """
    Show preprocessed image tensor with heatmap overlay.
    """
    img = image_tensor.detach().cpu().permute(1, 2, 0).numpy()
    img = img - img.min()
    if img.max() > 0:
        img = img / img.max()

    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.imshow(heatmap, cmap="jet", alpha=alpha)
    plt.axis("off")
    plt.show()