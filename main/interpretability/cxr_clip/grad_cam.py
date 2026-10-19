from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def _get_target_layer(model):
    """
    CXR-CLIP ResNet50 visual encoder:
        model.image_encoder.resnet -> torchvision ResNet
    """
    return model.image_encoder.resnet.layer4[-1]


@torch.no_grad()
def _get_label_text_embeddings(
    pipeline: Dict,
    label: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns mean positive and mean negative text embeddings for one label.
    Shape: [1, D], normalized.
    """
    model = pipeline["model"]
    device = next(model.parameters()).device

    text_cache = pipeline.get("text_embedding_cache", None)
    if text_cache is not None and label in text_cache:
        pos_emb = text_cache[label]["pos"]
        neg_emb = text_cache[label]["neg"]
    else:
        from pipelines.cxr_clip_pipeline import get_text_embeddings_for_label
        embs = get_text_embeddings_for_label(pipeline, label)
        pos_emb = embs["pos"]
        neg_emb = embs["neg"]

    pos_emb = F.normalize(pos_emb, dim=-1).mean(dim=0, keepdim=True).to(device)
    neg_emb = F.normalize(neg_emb, dim=-1).mean(dim=0, keepdim=True).to(device)
    return pos_emb, neg_emb


def _forward_to_spatial_features(model, image_tensor: torch.Tensor) -> torch.Tensor:
    """
    Forward image through ResNet up to layer4 output.
    Returns [B, C, H, W]
    """
    backbone = model.image_encoder.resnet

    x = backbone.conv1(image_tensor)
    x = backbone.bn1(x)
    x = backbone.relu(x)
    x = backbone.maxpool(x)

    x = backbone.layer1(x)
    x = backbone.layer2(x)
    x = backbone.layer3(x)
    x = backbone.layer4(x)
    return x


def _global_embed_from_spatial(model, spatial_feat: torch.Tensor) -> torch.Tensor:
    """
    Convert spatial layer4 features into the same global image embedding space used by CXR-CLIP.
    spatial_feat: [B, C, H, W]
    returns: [B, D]
    """
    pooled = F.adaptive_avg_pool2d(spatial_feat, output_size=(1, 1)).flatten(1)  # [B, C]
    img_emb = model.image_projection(pooled) if getattr(model, "projection", False) else pooled
    img_emb = F.normalize(img_emb, dim=-1)
    return img_emb


def _target_score_from_spatial(
    pipeline: Dict,
    spatial_feat: torch.Tensor,
    label: str,
    use_margin: bool = True,
) -> torch.Tensor:
    """
    Compute the CXR-CLIP target score from spatial image features.
    """
    model = pipeline["model"]
    pos_emb, neg_emb = _get_label_text_embeddings(pipeline, label)

    img_emb = _global_embed_from_spatial(model, spatial_feat)  # [B, D]
    pos_score = (img_emb @ pos_emb.T).squeeze(-1)              # [B]

    if use_margin:
        neg_score = (img_emb @ neg_emb.T).squeeze(-1)          # [B]
        score = pos_score - neg_score
    else:
        score = pos_score

    return score.sum()


def _normalize_cam(cam: np.ndarray) -> np.ndarray:
    cam = cam.astype(np.float32)
    if cam.max() > cam.min():
        cam = (cam - cam.min()) / (cam.max() - cam.min())
    else:
        cam = np.zeros_like(cam, dtype=np.float32)
    return cam


def generate_gradcam(
    pipeline: Dict,
    image_tensor: torch.Tensor,
    label: str,
    use_margin: bool = True,
) -> np.ndarray:
    """
    Standard Grad-CAM for CXR-CLIP.
    Returns heatmap [H, W] normalized to [0,1].
    """
    model = pipeline["model"]
    device = next(model.parameters()).device
    model.eval()

    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)

    activations = []
    gradients = []

    target_layer = _get_target_layer(model)

    def forward_hook(module, inp, out):
        activations.append(out)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    model.zero_grad(set_to_none=True)

    spatial_feat = _forward_to_spatial_features(model, image_tensor)
    score = _target_score_from_spatial(
        pipeline=pipeline,
        spatial_feat=spatial_feat,
        label=label,
        use_margin=use_margin,
    )
    score.backward()

    fh.remove()
    bh.remove()

    act = activations[0]   # [B, C, Hf, Wf]
    grad = gradients[0]    # [B, C, Hf, Wf]

    weights = grad.mean(dim=(2, 3), keepdim=True)            # [B, C, 1, 1]
    cam = (weights * act).sum(dim=1, keepdim=True)           # [B, 1, Hf, Wf]
    cam = F.relu(cam)

    cam = F.interpolate(
        cam,
        size=image_tensor.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )

    cam = cam[0, 0].detach().cpu().numpy()
    return _normalize_cam(cam)


def generate_gradcam_plus_plus(
    pipeline: Dict,
    image_tensor: torch.Tensor,
    label: str,
    use_margin: bool = True,
) -> np.ndarray:
    """
    Grad-CAM++ for CXR-CLIP.
    Returns heatmap [H, W] normalized to [0,1].
    """
    model = pipeline["model"]
    device = next(model.parameters()).device
    model.eval()

    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)

    activations = []
    gradients = []

    target_layer = _get_target_layer(model)

    def forward_hook(module, inp, out):
        activations.append(out)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    model.zero_grad(set_to_none=True)

    spatial_feat = _forward_to_spatial_features(model, image_tensor)
    score = _target_score_from_spatial(
        pipeline=pipeline,
        spatial_feat=spatial_feat,
        label=label,
        use_margin=use_margin,
    )
    score.backward()

    fh.remove()
    bh.remove()

    act = activations[0]   # [B, C, Hf, Wf]
    grad = gradients[0]    # [B, C, Hf, Wf]

    grad2 = grad ** 2
    grad3 = grad ** 3

    sum_act_grad3 = (act * grad3).sum(dim=(2, 3), keepdim=True)  # [B,C,1,1]
    eps = 1e-8

    alpha = grad2 / (2.0 * grad2 + sum_act_grad3 + eps)
    positive_grad = F.relu(grad)
    weights = (alpha * positive_grad).sum(dim=(2, 3), keepdim=True)  # [B,C,1,1]

    cam = (weights * act).sum(dim=1, keepdim=True)  # [B,1,Hf,Wf]
    cam = F.relu(cam)

    cam = F.interpolate(
        cam,
        size=image_tensor.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )

    cam = cam[0, 0].detach().cpu().numpy()
    return _normalize_cam(cam)


def show_cam_overlay(
    image_tensor: torch.Tensor,
    heatmap: np.ndarray,
    alpha: float = 0.35,
    cmap: str = "magma",
    title: str | None = None,
):
    img = image_tensor.detach().cpu().permute(1, 2, 0).numpy()
    img = img - img.min()
    if img.max() > 0:
        img = img / img.max()

    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.imshow(heatmap, cmap=cmap, alpha=alpha)
    if title is not None:
        plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()