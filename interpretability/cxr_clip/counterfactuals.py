# ============================================================
# interpretability/cxr_clip/counterfactuals.py
# ============================================================

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import pandas as pd
import torch
import torch.nn.functional as F

from models.cxr_clip.load import encode_image, encode_text, score_prompts
from pipelines.cxr_clip_pipeline import score_image_tensor

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# 1. Textual Counterfactuals (prompt modifications)
# ============================================================

def evaluate_prompt_counterfactual(
    pipeline: Dict[str, Any],
    image_tensor: torch.Tensor,
    label: str,
    counterfactual_prompts: Dict[str, List[str]],
) -> Dict[str, Any]:
    """
    Compare original prompt pair vs counterfactual prompt pair for one label.
    
    Args:
        pipeline: CXR-CLIP pipeline
        image_tensor: [3, H, W]
        label: pathology label
        counterfactual_prompts: {"pos": [...], "neg": [...]}
    """
    model = pipeline["model"]
    tokenizer = pipeline["tokenizer"]

    # Original scores
    orig_prompts = pipeline["prompts"][label]
    orig_pos = score_prompts(model, tokenizer, image_tensor, orig_prompts["pos"]).mean().item()
    orig_neg = score_prompts(model, tokenizer, image_tensor, orig_prompts["neg"]).mean().item()
    orig_margin = orig_pos - orig_neg

    # Counterfactual scores
    cf_pos = score_prompts(model, tokenizer, image_tensor, counterfactual_prompts["pos"]).mean().item()
    cf_neg = score_prompts(model, tokenizer, image_tensor, counterfactual_prompts["neg"]).mean().item()
    cf_margin = cf_pos - cf_neg

    return {
        "label": label,
        "original_pos": orig_pos,
        "original_neg": orig_neg,
        "original_margin": orig_margin,
        "cf_pos": cf_pos,
        "cf_neg": cf_neg,
        "cf_margin": cf_margin,
        "margin_drop": orig_margin - cf_margin,
    }


def evaluate_negation_counterfactual(
    pipeline: Dict[str, Any],
    image_tensor: torch.Tensor,
    label: str,
) -> Dict[str, Any]:
    """
    Swap positive and negative prompts to test if the model
    is sensitive to semantic meaning vs surface features.
    """
    orig_prompts = pipeline["prompts"][label]

    # Counterfactual: swap pos and neg
    swapped = {
        "pos": orig_prompts["neg"],
        "neg": orig_prompts["pos"],
    }

    return evaluate_prompt_counterfactual(
        pipeline=pipeline,
        image_tensor=image_tensor,
        label=label,
        counterfactual_prompts=swapped,
    )


def evaluate_generic_prompt_counterfactual(
    pipeline: Dict[str, Any],
    image_tensor: torch.Tensor,
    label: str,
) -> Dict[str, Any]:
    """
    Replace disease-specific prompts with generic ones
    to test concept specificity.
    """
    generic = {
        "pos": ["An abnormality is present in the chest X-ray."],
        "neg": ["The chest X-ray appears normal."],
    }

    return evaluate_prompt_counterfactual(
        pipeline=pipeline,
        image_tensor=image_tensor,
        label=label,
        counterfactual_prompts=generic,
    )


# ============================================================
# 2. Latent-Space Counterfactual Edits
# ============================================================

def cxr_clip_latent_counterfactual(
    image_embedding: torch.Tensor,
    pipeline: Dict[str, Any],
    label: str,
    concept_vector: torch.Tensor,
    epsilon: float = 0.10,
) -> Dict[str, Any]:
    """
    Edit image embedding along concept direction and measure
    margin change.
    """
    e = F.normalize(image_embedding.float(), dim=0)
    c = F.normalize(concept_vector.float(), dim=0)

    toward = F.normalize(e + epsilon * c, dim=0)
    away = F.normalize(e - epsilon * c, dim=0)

    from pipelines.cxr_clip_pipeline import score_image_embedding

    orig_results = score_image_embedding(pipeline, e)
    toward_results = score_image_embedding(pipeline, toward)
    away_results = score_image_embedding(pipeline, away)

    return {
        "label": label,
        "epsilon": epsilon,
        "original_margin": orig_results[label]["margin"],
        "toward_margin": toward_results[label]["margin"],
        "away_margin": away_results[label]["margin"],
        "toward_delta": toward_results[label]["margin"] - orig_results[label]["margin"],
        "away_delta": away_results[label]["margin"] - orig_results[label]["margin"],
        "original_projection": float(torch.dot(e, c).item()),
        "toward_projection": float(torch.dot(toward, c).item()),
        "away_projection": float(torch.dot(away, c).item()),
    }


# ============================================================
# 3. Confidence Drop (masking-based)
# ============================================================

def evaluate_masking_counterfactual(
    pipeline: Dict[str, Any],
    image_tensor: torch.Tensor,
    label: str,
    heatmap,
    fraction: float = 0.2,
    fill_value: float = 0.0,
) -> Dict[str, Any]:
    """
    Mask the top-k% most relevant pixels and measure margin drop.
    """
    import numpy as np

    if isinstance(heatmap, torch.Tensor):
        heatmap_np = heatmap.detach().cpu().numpy()
    else:
        heatmap_np = heatmap

    # Original score
    orig_results = score_image_tensor(pipeline, image_tensor)
    orig_margin = orig_results[label]["margin"]

    # Build mask
    flat = heatmap_np.flatten()
    k = int(fraction * flat.size)
    if k <= 0:
        return {
            "label": label,
            "original_margin": orig_margin,
            "masked_margin": orig_margin,
            "margin_drop": 0.0,
            "fraction": fraction,
        }

    threshold = np.partition(flat, -k)[-k]
    mask = torch.tensor(heatmap_np >= threshold, dtype=image_tensor.dtype,
                        device=image_tensor.device)

    if image_tensor.ndim == 3:
        mask = mask.unsqueeze(0)

    masked_image = image_tensor * (1 - mask) + fill_value * mask
    masked_results = score_image_tensor(pipeline, masked_image)
    masked_margin = masked_results[label]["margin"]

    return {
        "label": label,
        "original_margin": orig_margin,
        "masked_margin": masked_margin,
        "margin_drop": orig_margin - masked_margin,
        "fraction": fraction,
    }


# ============================================================
# 4. Multi-sample runners
# ============================================================

def run_negation_counterfactuals(
    pipeline: Dict[str, Any],
    test_df,
    cxr_clip_samples: List[Dict],
    pathologies: List[str],
    max_per_pathology: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    from data_loader import CXRClipInferenceDataset
    rng = random.Random(seed)
    rows = []

    for label in pathologies:
        pos_samples = [s for s in cxr_clip_samples if s["labels"].get(label, 0) == 1.0]
        selected = rng.sample(pos_samples, min(max_per_pathology, len(pos_samples)))

        for s in selected:
            img_path = s["metadata"]["image_path"]
            sample_df = test_df[test_df["image_path"] == img_path].reset_index(drop=True)
            if len(sample_df) == 0:
                continue

            dataset = CXRClipInferenceDataset(sample_df, transform=pipeline["preprocess"])
            sample = dataset[0]

            cf = evaluate_negation_counterfactual(pipeline, sample["image"], label)
            cf["subject_id"] = s["metadata"].get("subject_id")
            cf["image_path"] = img_path
            rows.append(cf)

    return pd.DataFrame(rows)


def run_generic_prompt_counterfactuals(
    pipeline: Dict[str, Any],
    test_df,
    cxr_clip_samples: List[Dict],
    pathologies: List[str],
    max_per_pathology: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    from data_loader import CXRClipInferenceDataset
    rng = random.Random(seed)
    rows = []

    for label in pathologies:
        pos_samples = [s for s in cxr_clip_samples if s["labels"].get(label, 0) == 1.0]
        selected = rng.sample(pos_samples, min(max_per_pathology, len(pos_samples)))

        for s in selected:
            img_path = s["metadata"]["image_path"]
            sample_df = test_df[test_df["image_path"] == img_path].reset_index(drop=True)
            if len(sample_df) == 0:
                continue

            dataset = CXRClipInferenceDataset(sample_df, transform=pipeline["preprocess"])
            sample = dataset[0]

            cf = evaluate_generic_prompt_counterfactual(pipeline, sample["image"], label)
            cf["subject_id"] = s["metadata"].get("subject_id")
            cf["image_path"] = img_path
            rows.append(cf)

    return pd.DataFrame(rows)


def run_cxr_clip_latent_counterfactuals(
    pipeline: Dict[str, Any],
    cxr_clip_samples: List[Dict],
    concept_vectors: Dict[str, torch.Tensor],
    pathologies: List[str],
    max_per_pathology: int = 50,
    epsilon: float = 0.10,
    seed: int = 42,
) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []

    for label in pathologies:
        if label not in concept_vectors:
            continue

        pos_samples = [s for s in cxr_clip_samples if s["labels"].get(label, 0) == 1.0]
        selected = rng.sample(pos_samples, min(max_per_pathology, len(pos_samples)))

        for s in selected:
            cf = cxr_clip_latent_counterfactual(
                image_embedding=s["image_emb"],
                pipeline=pipeline,
                label=label,
                concept_vector=concept_vectors[label],
                epsilon=epsilon,
            )
            cf["subject_id"] = s["metadata"].get("subject_id")
            rows.append(cf)

    return pd.DataFrame(rows)


def plot_cxr_clip_negation_counterfactuals(df, save_path=None):
    labels = df["label"].unique()
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("white")

    x = np.arange(len(labels))
    width = 0.35

    orig = [df[df["label"] == l]["original_margin"].mean() for l in labels]
    cf = [df[df["label"] == l]["cf_margin"].mean() for l in labels]

    bars1 = ax.bar(x - width/2, orig, width, label="Original Prompts",
                   color="#0077B6", edgecolor="white", linewidth=0.8)
    bars2 = ax.bar(x + width/2, cf, width, label="Swapped Prompts",
                   color="#E63946", edgecolor="white", linewidth=0.8)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.002,
                    f"{h:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Mean Margin", fontsize=12)
    ax.set_title("CXR-CLIP: Negation Counterfactuals\n(Swapping Positive ↔ Negative Prompts)",
                 fontsize=14, fontweight="bold")
    ax.legend(frameon=False, fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#E0E0E0", linewidth=0.6)
    ax.set_axisbelow(True)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=250, bbox_inches="tight", facecolor="white")
        print(f"Saved: {save_path}")
    plt.show()


def plot_cxr_clip_generic_counterfactuals(df, save_path=None):
    labels = df["label"].unique()
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("white")

    x = np.arange(len(labels))
    width = 0.35

    orig = [df[df["label"] == l]["original_margin"].mean() for l in labels]
    generic = [df[df["label"] == l]["cf_margin"].mean() for l in labels]

    bars1 = ax.bar(x - width/2, orig, width, label="Disease-Specific Prompts",
                   color="#0077B6", edgecolor="white", linewidth=0.8)
    bars2 = ax.bar(x + width/2, generic, width, label="Generic Prompts",
                   color="#F4A261", edgecolor="white", linewidth=0.8)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.002,
                    f"{h:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Mean Margin", fontsize=12)
    ax.set_title("CXR-CLIP: Generic vs Disease-Specific Prompts",
                 fontsize=14, fontweight="bold")
    ax.legend(frameon=False, fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#E0E0E0", linewidth=0.6)
    ax.set_axisbelow(True)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=250, bbox_inches="tight", facecolor="white")
        print(f"Saved: {save_path}")
    plt.show()


def plot_cxr_clip_latent_counterfactuals(df, save_path=None):
    labels = df["label"].unique()
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("white")

    x = np.arange(len(labels))
    width = 0.25

    orig = [df[df["label"] == l]["original_margin"].mean() for l in labels]
    toward = [df[df["label"] == l]["toward_margin"].mean() for l in labels]
    away = [df[df["label"] == l]["away_margin"].mean() for l in labels]

    bars1 = ax.bar(x - width, orig, width, label="Original",
                   color="#0077B6", edgecolor="white", linewidth=0.8)
    bars2 = ax.bar(x, toward, width, label="Toward Concept",
                   color="#2A9D8F", edgecolor="white", linewidth=0.8)
    bars3 = ax.bar(x + width, away, width, label="Away from Concept",
                   color="#E63946", edgecolor="white", linewidth=0.8)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.001,
                    f"{h:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Mean Margin", fontsize=12)
    ax.set_title("CXR-CLIP: Latent-Space Counterfactual Edits",
                 fontsize=14, fontweight="bold")
    ax.legend(frameon=False, fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#E0E0E0", linewidth=0.6)
    ax.set_axisbelow(True)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=250, bbox_inches="tight", facecolor="white")
        print(f"Saved: {save_path}")
    plt.show()