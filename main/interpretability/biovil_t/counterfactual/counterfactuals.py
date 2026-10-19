from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence
import random

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from preprocessors.biovil_t import load_biovil_preprocessor
from models.biovil_t.load import load_biovil_t

from interpretability.biovil_t.counterfactual.attention_control import (
    get_active_attention_suppressor,
    use_attention_suppressor,
)

# =========================================================
# Init helpers
# =========================================================

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def init_biovilt_counterfactuals(
    model_name: str = "microsoft/BiomedVLP-BioViL-T",
    device: str | None = None,
) -> Dict[str, Any]:
    if device is None:
        device = get_device()

    model = load_biovil_t(model_name=model_name, device=device)
    model.eval()

    tokenizer = load_biovil_preprocessor(model_name=model_name, mode="text")
    image_processor = load_biovil_preprocessor(model_name=model_name, mode="image")

    return {
        "model": model,
        "tokenizer": tokenizer,
        "image_processor": image_processor,
        "device": device,
        "model_name": model_name,
    }


def tokenize_text(
    tokenizer,
    text: str,
    device: str | torch.device,
) -> Dict[str, torch.Tensor]:
    enc = tokenizer(
        text or "",
        return_tensors="pt",
        truncation=True,
        padding=True,
    )
    return {
        "input_ids": enc["input_ids"].to(device),
        "attention_mask": enc["attention_mask"].to(device),
    }


# =========================================================
# Sample preparation
# =========================================================

def build_matched_single_samples(single_samples: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    matched = []

    for s in single_samples:
        meta = s.get("metadata", {})

        if s.get("image_emb") is None:
            continue
        if s.get("text_emb") is None:
            continue
        if meta.get("image_path") is None:
            continue

        image_path = Path(meta["image_path"])
        if not image_path.exists():
            continue

        matched.append(s)

    return matched


def build_matched_temporal_samples(temporal_samples: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    matched = []

    for s in temporal_samples:
        meta = s.get("metadata", {})

        if s.get("text_emb") is None:
            continue
        if meta.get("current_image_path") is None:
            continue
        if meta.get("prior_image_path") is None:
            continue

        current_path = Path(meta["current_image_path"])
        prior_path = Path(meta["prior_image_path"])

        if not current_path.exists():
            continue
        if not prior_path.exists():
            continue

        matched.append(s)

    return matched


def choose_random_samples(
    samples: Sequence[Dict[str, Any]],
    num_samples: int = 10,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    samples = list(samples)

    if len(samples) == 0:
        raise ValueError("No samples available.")

    return rng.sample(samples, min(num_samples, len(samples)))


def choose_mismatched_prior(
    temporal_samples: Sequence[Dict[str, Any]],
    current_subject_id,
) -> Dict[str, Any]:
    for s in temporal_samples:
        meta = s["metadata"]
        if meta.get("subject_id") != current_subject_id:
            return s
    raise ValueError("Could not find mismatched prior from a different subject.")


# =========================================================
# I/O helpers
# =========================================================



def load_grayscale_image_tensor(image_path, image_processor):
    """
    Returns a preprocessed tensor [3, H, W] in the format expected by BioViL-T.
    """
    image = Image.open(image_path).convert("L")
    out = image_processor(image)
    if isinstance(out, dict):
        if "pixel_values" in out:
            x = out["pixel_values"]
            if x.ndim == 4:
                x = x[0]
            return x
        raise KeyError(f"Expected 'pixel_values' in processor output, got keys: {list(out.keys())}")
    return out


def get_single_report_text(sample: Dict[str, Any]) -> str:
    return sample.get("metadata", {}).get("report_text", "") or ""


def get_temporal_report_text(sample: Dict[str, Any]) -> str:
    return sample.get("metadata", {}).get("current_report_text", "") or ""


# =========================================================
# Similarity scoring
# =========================================================

@torch.no_grad()
def biovilt_single_similarity_score(
    model,
    pixel_values: torch.Tensor,      # [B, 3, H, W]
    input_ids: torch.Tensor,         # [B, T]
    attention_mask: torch.Tensor,    # [B, T]
) -> torch.Tensor:
    image_out = model.image_model(
        current_image=pixel_values,
        previous_image=None,
    )
    image_emb = F.normalize(image_out.projected_global_embedding, dim=-1)

    text_emb = model.encode_text(input_ids, attention_mask)
    text_emb = F.normalize(text_emb, dim=-1)

    return torch.sum(image_emb * text_emb, dim=-1)


@torch.no_grad()
def biovilt_temporal_similarity_score(
    model,
    current_pixel_values: torch.Tensor,
    prior_pixel_values: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    image_out = model.image_model(
        current_image=current_pixel_values,
        previous_image=prior_pixel_values,
    )
    temporal_emb = F.normalize(image_out.projected_global_embedding, dim=-1)

    text_emb = model.encode_text(input_ids, attention_mask)
    text_emb = F.normalize(text_emb, dim=-1)

    return torch.sum(temporal_emb * text_emb, dim=-1)


# =========================================================
# True attention suppression controller
# =========================================================

@dataclass
class AttentionSuppressionConfig:
    target: str = "topk"  # "topk", "heads", "temporal_quadrant"
    layer_indices: Optional[List[int]] = None
    head_indices: Optional[List[int]] = None
    renormalize: bool = True

    # for target="topk"
    topk_fraction: float = 0.10

    # for target="temporal_quadrant"
    current_length: int = 196
    previous_length: int = 196
    quadrant: str = "current_to_current"

    # how strong
    mode: str = "zero"     # "zero" or "scale"
    scale: float = 0.0


def _renormalize_attention(attn_probs: torch.Tensor) -> torch.Tensor:
    return attn_probs / (attn_probs.sum(dim=-1, keepdim=True) + 1e-8)


def _build_topk_mask(attn_probs: torch.Tensor, topk_fraction: float) -> torch.Tensor:
    """
    attn_probs: [B, H, N, N]
    suppress top-k attention values per row
    """
    if not (0.0 < topk_fraction <= 1.0):
        raise ValueError("topk_fraction must be in (0, 1].")

    B, H, N, _ = attn_probs.shape
    k = max(1, int(topk_fraction * N))

    idx = torch.topk(attn_probs, k=k, dim=-1).indices
    mask = torch.zeros_like(attn_probs)
    mask.scatter_(-1, idx, 1.0)
    return mask


def _build_head_mask(attn_probs: torch.Tensor, head_indices: Sequence[int]) -> torch.Tensor:
    mask = torch.zeros_like(attn_probs)
    mask[:, list(head_indices), :, :] = 1.0
    return mask


def _build_temporal_quadrant_mask(
    attn_probs: torch.Tensor,
    current_length: int,
    previous_length: int,
    quadrant: str,
) -> torch.Tensor:
    """
    attn_probs: [B, H, N, N], where N = current_length + previous_length
    """
    B, H, N, _ = attn_probs.shape
    expected = current_length + previous_length
    if N != expected:
        raise ValueError(f"Expected {expected} tokens, got {N}")

    mask = torch.zeros_like(attn_probs)

    cur0, cur1 = 0, current_length
    prev0, prev1 = current_length, current_length + previous_length

    if quadrant == "current_to_current":
        mask[:, :, cur0:cur1, cur0:cur1] = 1.0
    elif quadrant == "current_to_previous":
        mask[:, :, cur0:cur1, prev0:prev1] = 1.0
    elif quadrant == "previous_to_current":
        mask[:, :, prev0:prev1, cur0:cur1] = 1.0
    elif quadrant == "previous_to_previous":
        mask[:, :, prev0:prev1, prev0:prev1] = 1.0
    else:
        raise ValueError(f"Unknown temporal quadrant: {quadrant}")

    return mask


def make_attention_suppressor(config: AttentionSuppressionConfig) -> Callable:
    """
    Returns a callable:
        suppressor(attn_probs, layer_idx) -> modified_attn_probs
    """

    def suppressor(attn_probs: torch.Tensor, layer_idx: Optional[int] = None) -> torch.Tensor:
        out = attn_probs

        if config.layer_indices is not None:
            if layer_idx is None or layer_idx not in config.layer_indices:
                return out

        if config.target == "topk":
            mask = _build_topk_mask(out, config.topk_fraction)

        elif config.target == "heads":
            if config.head_indices is None:
                raise ValueError("head_indices must be provided when target='heads'")
            mask = _build_head_mask(out, config.head_indices)

        elif config.target == "temporal_quadrant":
            mask = _build_temporal_quadrant_mask(
                out,
                current_length=config.current_length,
                previous_length=config.previous_length,
                quadrant=config.quadrant,
            )
        else:
            raise ValueError(f"Unknown suppression target: {config.target}")

        if config.mode == "zero":
            out = out * (1.0 - mask)
        elif config.mode == "scale":
            out = out * (1.0 - mask) + out * mask * config.scale
        else:
            raise ValueError(f"Unknown suppression mode: {config.mode}")

        if config.renormalize:
            out = _renormalize_attention(out)

        return out

    return suppressor

@torch.no_grad()
def get_temporal_attentions_with_optional_suppression(
    model,
    current_image_tensor,   # [3, H, W]
    prior_image_tensor,     # [3, H, W]
    suppressor=None,
):
    device = next(model.parameters()).device

    current_b = current_image_tensor.unsqueeze(0).to(device)
    prior_b = prior_image_tensor.unsqueeze(0).to(device)

    with use_attention_suppressor(suppressor):
        patch_fused, pooled_emb, attentions = model.image_model.encoder(
            current_image=current_b,
            previous_image=prior_b,
            return_patch_embeddings=True,
            return_attentions=True,
        )

    return patch_fused, attentions

# =========================================================
# True attention suppression scoring
# =========================================================

@torch.no_grad()
def biovilt_single_similarity_score_with_attention_suppression(
    model,
    pixel_values: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    suppressor: Optional[Callable] = None,
) -> torch.Tensor:
    with use_attention_suppressor(suppressor):
        return biovilt_single_similarity_score(
            model=model,
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )


@torch.no_grad()
def biovilt_temporal_similarity_score_with_attention_suppression(
    model,
    current_pixel_values: torch.Tensor,
    prior_pixel_values: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    suppressor: Optional[Callable] = None,
) -> torch.Tensor:
    with use_attention_suppressor(suppressor):
        return biovilt_temporal_similarity_score(
            model=model,
            current_pixel_values=current_pixel_values,
            prior_pixel_values=prior_pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )


def evaluate_true_attention_suppression_counterfactual_single(
    model,
    image_tensor: torch.Tensor,      # [3, H, W]
    input_ids: torch.Tensor,         # [1, T]
    attention_mask: torch.Tensor,    # [1, T]
    suppression_config: AttentionSuppressionConfig,
) -> Dict[str, Any]:
    model.eval()
    device = next(model.parameters()).device

    pixel_values = image_tensor.unsqueeze(0).to(device)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    suppressor = make_attention_suppressor(suppression_config)

    s_orig = biovilt_single_similarity_score_with_attention_suppression(
        model=model,
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        suppressor=None,
    )[0].item()

    s_supp = biovilt_single_similarity_score_with_attention_suppression(
        model=model,
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        suppressor=suppressor,
    )[0].item()

    return {
        "original_score": s_orig,
        "suppressed_score": s_supp,
        "score_drop": s_orig - s_supp,
        "suppression_target": suppression_config.target,
        "suppression_mode": suppression_config.mode,
    }


def evaluate_true_attention_suppression_counterfactual_temporal(
    model,
    current_image: torch.Tensor,
    prior_image: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    suppression_config: AttentionSuppressionConfig,
) -> Dict[str, Any]:
    model.eval()
    device = next(model.parameters()).device

    current_b = current_image.unsqueeze(0).to(device)
    prior_b = prior_image.unsqueeze(0).to(device)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    suppressor = make_attention_suppressor(suppression_config)

    s_orig = biovilt_temporal_similarity_score_with_attention_suppression(
        model=model,
        current_pixel_values=current_b,
        prior_pixel_values=prior_b,
        input_ids=input_ids,
        attention_mask=attention_mask,
        suppressor=None,
    )[0].item()

    s_supp = biovilt_temporal_similarity_score_with_attention_suppression(
        model=model,
        current_pixel_values=current_b,
        prior_pixel_values=prior_b,
        input_ids=input_ids,
        attention_mask=attention_mask,
        suppressor=suppressor,
    )[0].item()

    return {
        "original_score": s_orig,
        "suppressed_score": s_supp,
        "score_drop": s_orig - s_supp,
        "suppression_target": suppression_config.target,
        "suppression_mode": suppression_config.mode,
    }


# =========================================================
# Textual counterfactuals
# =========================================================

def evaluate_textual_counterfactual(
    model,
    tokenizer,
    pixel_values: torch.Tensor,      # [1, 3, H, W]
    original_text: str,
    counterfactual_text: str,
) -> Dict[str, Any]:
    model.eval()
    device = next(model.parameters()).device
    pixel_values = pixel_values.to(device)

    enc_orig = tokenize_text(tokenizer, original_text, device)
    enc_cf = tokenize_text(tokenizer, counterfactual_text, device)

    s_orig = biovilt_single_similarity_score(
        model,
        pixel_values,
        enc_orig["input_ids"],
        enc_orig["attention_mask"],
    )[0].item()

    s_cf = biovilt_single_similarity_score(
        model,
        pixel_values,
        enc_cf["input_ids"],
        enc_cf["attention_mask"],
    )[0].item()

    return {
        "original_text": original_text,
        "counterfactual_text": counterfactual_text,
        "original_score": s_orig,
        "counterfactual_score": s_cf,
        "score_drop": s_orig - s_cf,
    }


# =========================================================
# Temporal counterfactual variations
# =========================================================

def evaluate_temporal_counterfactual(
    model,
    current_image: torch.Tensor,
    prior_image: torch.Tensor,
    mismatched_prior_image: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Dict[str, Any]:
    model.eval()
    device = next(model.parameters()).device

    current_b = current_image.unsqueeze(0).to(device)
    prior_b = prior_image.unsqueeze(0).to(device)
    mismatch_b = mismatched_prior_image.unsqueeze(0).to(device)

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    s_temporal = biovilt_temporal_similarity_score(
        model, current_b, prior_b, input_ids, attention_mask
    )[0].item()

    s_mismatch = biovilt_temporal_similarity_score(
        model, current_b, mismatch_b, input_ids, attention_mask
    )[0].item()

    s_current_only = biovilt_single_similarity_score(
        model, current_b, input_ids, attention_mask
    )[0].item()

    return {
        "temporal_score": s_temporal,
        "mismatched_prior_score": s_mismatch,
        "current_only_score": s_current_only,
        "drop_with_mismatch": s_temporal - s_mismatch,
        "drop_without_prior": s_temporal - s_current_only,
    }


# =========================================================
# Latent-space counterfactual edits
# =========================================================

def latent_counterfactual_edit(
    embedding: torch.Tensor,
    concept_vector: torch.Tensor,
    epsilon: float = 0.10,
) -> Dict[str, torch.Tensor]:
    e = F.normalize(embedding.float(), dim=0)
    c = F.normalize(concept_vector.float(), dim=0)

    plus = F.normalize(e + epsilon * c, dim=0)
    minus = F.normalize(e - epsilon * c, dim=0)

    return {
        "original": e,
        "toward_concept": plus,
        "away_from_concept": minus,
    }


def project_embedding_on_concept(
    embedding: torch.Tensor,
    concept_vector: torch.Tensor,
) -> float:
    e = F.normalize(embedding.float(), dim=0)
    c = F.normalize(concept_vector.float(), dim=0)
    return float(torch.dot(e, c).item())


def evaluate_latent_counterfactual_similarity(
    image_embedding: torch.Tensor,
    text_embedding: torch.Tensor,
    concept_vector: torch.Tensor,
    epsilon: float = 0.10,
) -> Dict[str, Any]:
    e_img = F.normalize(image_embedding.float(), dim=0)
    e_txt = F.normalize(text_embedding.float(), dim=0)
    c = F.normalize(concept_vector.float(), dim=0)

    edits = latent_counterfactual_edit(e_img, c, epsilon=epsilon)

    s_orig = torch.dot(edits["original"], e_txt).item()
    s_plus = torch.dot(edits["toward_concept"], e_txt).item()
    s_minus = torch.dot(edits["away_from_concept"], e_txt).item()

    return {
        "epsilon": epsilon,
        "original_similarity": s_orig,
        "toward_concept_similarity": s_plus,
        "away_from_concept_similarity": s_minus,
        "toward_delta": s_plus - s_orig,
        "away_delta": s_minus - s_orig,
        "original_concept_projection": project_embedding_on_concept(e_img, c),
        "toward_concept_projection": project_embedding_on_concept(edits["toward_concept"], c),
        "away_concept_projection": project_embedding_on_concept(edits["away_from_concept"], c),
        "edited_embeddings": edits,
    }


def evaluate_latent_counterfactual_batch(
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    concept_vector: torch.Tensor,
    epsilon: float = 0.10,
) -> Dict[str, Any]:
    image_embeddings = F.normalize(image_embeddings.float(), dim=-1)
    text_embeddings = F.normalize(text_embeddings.float(), dim=-1)
    c = F.normalize(concept_vector.float(), dim=0)

    toward = F.normalize(image_embeddings + epsilon * c.unsqueeze(0), dim=-1)
    away = F.normalize(image_embeddings - epsilon * c.unsqueeze(0), dim=-1)

    s_orig = torch.sum(image_embeddings * text_embeddings, dim=-1)
    s_toward = torch.sum(toward * text_embeddings, dim=-1)
    s_away = torch.sum(away * text_embeddings, dim=-1)

    proj_orig = torch.sum(image_embeddings * c.unsqueeze(0), dim=-1)
    proj_toward = torch.sum(toward * c.unsqueeze(0), dim=-1)
    proj_away = torch.sum(away * c.unsqueeze(0), dim=-1)

    return {
        "epsilon": epsilon,
        "mean_original_similarity": float(s_orig.mean().item()),
        "mean_toward_similarity": float(s_toward.mean().item()),
        "mean_away_similarity": float(s_away.mean().item()),
        "mean_toward_delta": float((s_toward - s_orig).mean().item()),
        "mean_away_delta": float((s_away - s_orig).mean().item()),
        "mean_original_projection": float(proj_orig.mean().item()),
        "mean_toward_projection": float(proj_toward.mean().item()),
        "mean_away_projection": float(proj_away.mean().item()),
        "all_original_similarity": s_orig.detach().cpu(),
        "all_toward_similarity": s_toward.detach().cpu(),
        "all_away_similarity": s_away.detach().cpu(),
    }


# =========================================================
# Multi-sample runners
# =========================================================

def run_true_attention_suppression_single_random(
    model,
    tokenizer,
    matched_single_samples: Sequence[Dict[str, Any]],
    suppression_config: AttentionSuppressionConfig,
    num_samples: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    chosen = choose_random_samples(matched_single_samples, num_samples=num_samples, seed=seed)
    rows = []

    for sample in chosen:
        image_tensor = load_grayscale_image_tensor(sample["metadata"]["image_path"])
        report_text = get_single_report_text(sample)
        enc = tokenize_text(tokenizer, report_text, next(model.parameters()).device)

        cf = evaluate_true_attention_suppression_counterfactual_single(
            model=model,
            image_tensor=image_tensor,
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            suppression_config=suppression_config,
        )

        rows.append({
            "subject_id": sample["metadata"].get("subject_id"),
            "study_id": sample["metadata"].get("study_id"),
            "dicom_id": sample["metadata"].get("dicom_id"),
            "original_score": cf["original_score"],
            "suppressed_score": cf["suppressed_score"],
            "score_drop": cf["score_drop"],
            "suppression_target": cf["suppression_target"],
            "suppression_mode": cf["suppression_mode"],
        })

    return pd.DataFrame(rows)


def run_true_attention_suppression_temporal_random(
    model,
    tokenizer,
    image_processor,
    matched_temporal_samples: Sequence[Dict[str, Any]],
    suppression_config: AttentionSuppressionConfig,
    num_samples: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    chosen = choose_random_samples(matched_temporal_samples, num_samples=num_samples, seed=seed)
    rows = []

    for sample in chosen:
        current_image_tensor = load_grayscale_image_tensor(
            sample["metadata"]["current_image_path"],
            image_processor
        )
        prior_image_tensor = load_grayscale_image_tensor(
            sample["metadata"]["prior_image_path"],
            image_processor
        )
        report_text = get_temporal_report_text(sample)
        enc = tokenize_text(tokenizer, report_text, next(model.parameters()).device)

        cf = evaluate_true_attention_suppression_counterfactual_temporal(
            model=model,
            current_image=current_image_tensor,
            prior_image=prior_image_tensor,
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            suppression_config=suppression_config,
        )

        rows.append({
            "subject_id": sample["metadata"].get("subject_id"),
            "current_study_id": sample["metadata"].get("current_study_id"),
            "current_dicom_id": sample["metadata"].get("current_dicom_id"),
            "original_score": cf["original_score"],
            "suppressed_score": cf["suppressed_score"],
            "score_drop": cf["score_drop"],
            "suppression_target": cf["suppression_target"],
            "suppression_mode": cf["suppression_mode"],
        })

    return pd.DataFrame(rows)


def run_textual_counterfactuals_random(
    model,
    tokenizer,
    image_processor,
    matched_single_samples: Sequence[Dict[str, Any]],
    counterfactual_fn: Callable[[str], str],
    num_samples: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    chosen = choose_random_samples(matched_single_samples, num_samples=num_samples, seed=seed)
    rows = []

    for sample in chosen:
        image_tensor = load_grayscale_image_tensor(
            sample["metadata"]["image_path"],
            image_processor=image_processor,
        )
        original_text = get_single_report_text(sample)
        counterfactual_text = counterfactual_fn(original_text)

        cf = evaluate_textual_counterfactual(
            model=model,
            tokenizer=tokenizer,
            pixel_values=image_tensor.unsqueeze(0).to(next(model.parameters()).device),
            original_text=original_text,
            counterfactual_text=counterfactual_text,
        )

        rows.append({
            "subject_id": sample["metadata"].get("subject_id"),
            "study_id": sample["metadata"].get("study_id"),
            "dicom_id": sample["metadata"].get("dicom_id"),
            "original_score": cf["original_score"],
            "counterfactual_score": cf["counterfactual_score"],
            "score_drop": cf["score_drop"],
        })

    return pd.DataFrame(rows)


def run_temporal_counterfactuals_random(
    model,
    tokenizer,
    image_processor,
    matched_temporal_samples: Sequence[Dict[str, Any]],
    num_samples: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    chosen = choose_random_samples(matched_temporal_samples, num_samples=num_samples, seed=seed)
    rows = []

    for sample in chosen:
        current_image_tensor = load_grayscale_image_tensor(
            sample["metadata"]["current_image_path"],
            image_processor=image_processor,
        )
        prior_image_tensor = load_grayscale_image_tensor(
            sample["metadata"]["prior_image_path"],
            image_processor=image_processor,
        )
        report_text = get_temporal_report_text(sample)

        enc = tokenize_text(tokenizer, report_text, next(model.parameters()).device)

        mismatch_sample = choose_mismatched_prior(
            temporal_samples=matched_temporal_samples,
            current_subject_id=sample["metadata"]["subject_id"],
        )
        mismatched_prior_tensor = load_grayscale_image_tensor(
            mismatch_sample["metadata"]["prior_image_path"],
            image_processor=image_processor,    
        )

        cf = evaluate_temporal_counterfactual(
            model=model,
            current_image=current_image_tensor,
            prior_image=prior_image_tensor,
            mismatched_prior_image=mismatched_prior_tensor,
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
        )

        rows.append({
            "subject_id": sample["metadata"].get("subject_id"),
            "current_study_id": sample["metadata"].get("current_study_id"),
            "current_dicom_id": sample["metadata"].get("current_dicom_id"),
            "temporal_score": cf["temporal_score"],
            "mismatched_prior_score": cf["mismatched_prior_score"],
            "current_only_score": cf["current_only_score"],
            "drop_with_mismatch": cf["drop_with_mismatch"],
            "drop_without_prior": cf["drop_without_prior"],
        })

    return pd.DataFrame(rows)


def run_latent_counterfactuals_random(
    matched_single_samples: Sequence[Dict[str, Any]],
    concept_vector: torch.Tensor,
    num_samples: int = 50,
    seed: int = 42,
    epsilon: float = 0.10,
) -> pd.DataFrame:
    chosen = choose_random_samples(matched_single_samples, num_samples=num_samples, seed=seed)
    rows = []

    for sample in chosen:
        cf = evaluate_latent_counterfactual_similarity(
            image_embedding=sample["image_emb"],
            text_embedding=sample["text_emb"],
            concept_vector=concept_vector,
            epsilon=epsilon,
        )

        rows.append({
            "subject_id": sample["metadata"].get("subject_id"),
            "study_id": sample["metadata"].get("study_id"),
            "dicom_id": sample["metadata"].get("dicom_id"),
            "original_similarity": cf["original_similarity"],
            "toward_concept_similarity": cf["toward_concept_similarity"],
            "away_concept_similarity": cf["away_from_concept_similarity"],
            "toward_delta": cf["toward_delta"],
            "away_delta": cf["away_delta"],
        })

    return pd.DataFrame(rows)


# =========================================================
# Summaries
# =========================================================

def summarize_counterfactual_df(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    return df[list(cols)].mean().to_frame(name="mean").T