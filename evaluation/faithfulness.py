# ============================================================
# evaluation/faithfulness.py
# Unified faithfulness evaluation for BioViL-T and CXR-CLIP
# ============================================================

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients


# ============================================================
# Colour palette
# ============================================================

PALETTE = {
    "BioViL-T": "#0077B6",
    "CXR-CLIP": "#E63946",
    "saliency": "#0077B6",
    "random": "#ADB5BD",
    "deletion": "#E63946",
    "insertion": "#2A9D8F",
    "grid": "#E0E0E0",
    "bg_header": "#1B2838",
    "bg_alt": "#F0F4F8",
}


# ============================================================
# Heatmap / mask helpers
# ============================================================

def normalize_heatmap(heatmap: torch.Tensor) -> torch.Tensor:
    heatmap = heatmap.float()
    heatmap = heatmap - heatmap.min()
    return heatmap / (heatmap.max() + 1e-8)


def resize_heatmap_to_image(heatmap: torch.Tensor, image_tensor: torch.Tensor) -> torch.Tensor:
    if heatmap.shape == image_tensor.shape[-2:]:
        return heatmap.float()
    return F.interpolate(
        heatmap.unsqueeze(0).unsqueeze(0).float(),
        size=image_tensor.shape[-2:],
        mode="bilinear", align_corners=False,
    )[0, 0]


def top_fraction_mask(heatmap: torch.Tensor, fraction: float = 0.2) -> torch.Tensor:
    flat = heatmap.reshape(-1)
    k = max(1, int(round(fraction * flat.numel())))
    _, idxs = torch.topk(flat, k=k)
    mask = torch.zeros_like(flat)
    mask[idxs] = 1.0
    return mask.view_as(heatmap)


def random_mask_like(image_tensor: torch.Tensor, fraction: float = 0.2) -> torch.Tensor:
    h, w = image_tensor.shape[-2:]
    flat = torch.zeros(h * w)
    k = max(1, int(round(fraction * flat.numel())))
    idx = torch.randperm(flat.numel())[:k]
    flat[idx] = 1.0
    return flat.view(h, w)


def apply_mask_to_image(image_tensor, mask_2d, fill_value=0.0):
    mask = mask_2d.unsqueeze(0).float()
    return image_tensor * (1.0 - mask) + fill_value * mask


def blur_image_region(image_tensor, mask_2d, kernel_size=21):
    mask = mask_2d.unsqueeze(0).float()
    blurred = F.avg_pool2d(
        image_tensor.unsqueeze(0), kernel_size=kernel_size,
        stride=1, padding=kernel_size // 2,
    )[0]
    return image_tensor * (1.0 - mask) + blurred * mask


def trapz_auc(x, y) -> float:
    return float(np.trapz(np.asarray(y, dtype=float), np.asarray(x, dtype=float)))


# ============================================================
# Scorer protocol: any model just needs a score function
#   scorer(image_tensor) -> float
# ============================================================

def make_biovilt_scorer(model, tokenizer, report_text: str):
    """Returns scorer(image_tensor) -> float for BioViL-T."""
    device = next(model.parameters()).device

    enc = tokenizer(report_text or "", return_tensors="pt", truncation=True, padding=True)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    @torch.no_grad()
    def scorer(image_tensor: torch.Tensor) -> float:
        x = image_tensor.unsqueeze(0).to(device) if image_tensor.ndim == 3 else image_tensor.to(device)
        image_out = model.image_model(current_image=x, previous_image=None)
        image_emb = F.normalize(image_out.projected_global_embedding, dim=-1)
        text_emb = model.encode_text(input_ids, attention_mask)
        text_emb = F.normalize(text_emb, dim=-1)
        return float(torch.sum(image_emb * text_emb, dim=-1)[0].item())

    return scorer, input_ids, attention_mask


def make_cxr_clip_scorer(pipeline, label: str):
    """Returns scorer(image_tensor) -> float for CXR-CLIP using margin."""
    from pipelines.cxr_clip_pipeline import score_image_tensor

    @torch.no_grad()
    def scorer(image_tensor: torch.Tensor) -> float:
        results = score_image_tensor(pipeline, image_tensor)
        return float(results[label]["margin"])

    return scorer


# ============================================================
# Integrated Gradients (both models)
# ============================================================

def _biovilt_ig_forward(model, input_ids, attention_mask):
    def forward_func(pixel_values):
        image_out = model.image_model(current_image=pixel_values, previous_image=None)
        image_emb = F.normalize(image_out.projected_global_embedding, dim=-1)
        text_emb = model.encode_text(input_ids, attention_mask)
        text_emb = F.normalize(text_emb, dim=-1)
        return torch.sum(image_emb * text_emb, dim=-1)
    return forward_func


def _cxr_clip_ig_forward(pipeline, label):
    model = pipeline["model"]
    device = next(model.parameters()).device
    text_cache = pipeline["text_embedding_cache"]
    pos_emb = text_cache[label]["pos"].to(device).detach()
    neg_emb = text_cache[label]["neg"].to(device).detach()

    # Check if resnet — resnet returns global features directly,
    # ViT returns sequence and needs [:, 0] for CLS token
    try:
        is_resnet = model.model_config["image_encoder"]["name"] == "resnet"
    except (AttributeError, KeyError):
        is_resnet = True  # your checkpoint uses resnet

    def forward_func(pixel_values):
        image_features = model.image_encoder(pixel_values)

        if not is_resnet:
            image_features = image_features[:, 0]

        image_features = model.image_projection(image_features)
        image_features = image_features / torch.norm(image_features, dim=1, keepdim=True)

        pos_score = (image_features @ pos_emb.T).mean(dim=-1)
        neg_score = (image_features @ neg_emb.T).mean(dim=-1)
        return pos_score - neg_score

    return forward_func


def compute_integrated_gradients(
    image_tensor: torch.Tensor,
    forward_func: Callable,
    device: str,
    baseline: torch.Tensor | None = None,
    n_steps: int = 50,
) -> Dict[str, Any]:
    x = image_tensor.unsqueeze(0).to(device) if image_tensor.ndim == 3 else image_tensor.to(device)

    if baseline is None:
        baseline = torch.zeros_like(x)
    elif baseline.ndim == 3:
        baseline = baseline.unsqueeze(0).to(device)

    ig = IntegratedGradients(forward_func)
    attributions, delta = ig.attribute(x, baselines=baseline, n_steps=n_steps, return_convergence_delta=True)

    heatmap = normalize_heatmap(attributions.abs().sum(dim=1)[0])

    return {
        "attributions": attributions.detach().cpu(),
        "heatmap": heatmap.detach().cpu(),
        "delta": delta.detach().cpu(),
    }


def compute_biovilt_integrated_gradients(
    model, image_tensor, tokenizer, report_text, n_steps=50,
):
    device = next(model.parameters()).device
    model.eval()
    enc = tokenizer(report_text or "", return_tensors="pt", truncation=True, padding=True)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    forward_func = _biovilt_ig_forward(model, input_ids, attention_mask)
    return compute_integrated_gradients(image_tensor, forward_func, device, n_steps=n_steps)


def compute_cxr_clip_integrated_gradients(
    pipeline, image_tensor, label, n_steps=50,
):
    device = next(pipeline["model"].parameters()).device
    forward_func = _cxr_clip_ig_forward(pipeline, label)
    return compute_integrated_gradients(image_tensor, forward_func, device, n_steps=n_steps)


# ============================================================
# Deletion / Insertion (model-agnostic via scorer)
# ============================================================

def evaluate_deletion_insertion(
    scorer: Callable[[torch.Tensor], float],
    image_tensor: torch.Tensor,
    heatmap: torch.Tensor,
    fractions=None,
    insertion_baseline: str = "zero",
) -> Dict[str, Any]:
    if fractions is None:
        fractions = np.linspace(0.0, 1.0, 11)

    heatmap = normalize_heatmap(resize_heatmap_to_image(heatmap, image_tensor))

    if insertion_baseline == "zero":
        base_image = torch.zeros_like(image_tensor)
    elif insertion_baseline == "blur":
        base_image = F.avg_pool2d(image_tensor.unsqueeze(0), kernel_size=21, stride=1, padding=10)[0]
    else:
        raise ValueError("insertion_baseline must be 'zero' or 'blur'")

    deletion_scores, insertion_scores = [], []

    for frac in fractions:
        mask = top_fraction_mask(heatmap, fraction=float(frac))

        deleted = apply_mask_to_image(image_tensor, mask, fill_value=0.0)
        deletion_scores.append(scorer(deleted))

        keep = mask.unsqueeze(0)
        inserted = base_image * (1.0 - keep) + image_tensor * keep
        insertion_scores.append(scorer(inserted))

    return {
        "fractions": list(fractions),
        "deletion_scores": deletion_scores,
        "insertion_scores": insertion_scores,
        "deletion_auc": trapz_auc(fractions, deletion_scores),
        "insertion_auc": trapz_auc(fractions, insertion_scores),
    }


# ============================================================
# Masking / Blurring vs Random (model-agnostic via scorer)
# ============================================================

def evaluate_masking_vs_random(
    scorer: Callable[[torch.Tensor], float],
    image_tensor: torch.Tensor,
    heatmap: torch.Tensor,
    fraction: float = 0.2,
    mode: str = "zero",
    n_random: int = 10,
) -> Dict[str, Any]:
    heatmap = normalize_heatmap(resize_heatmap_to_image(heatmap, image_tensor))
    original_score = scorer(image_tensor)

    sal_mask = top_fraction_mask(heatmap, fraction=fraction)
    if mode == "zero":
        sal_img = apply_mask_to_image(image_tensor, sal_mask)
    else:
        sal_img = blur_image_region(image_tensor, sal_mask)

    saliency_score = scorer(sal_img)

    random_scores = []
    for _ in range(n_random):
        rand_mask = random_mask_like(image_tensor, fraction=fraction)
        if mode == "zero":
            rand_img = apply_mask_to_image(image_tensor, rand_mask)
        else:
            rand_img = blur_image_region(image_tensor, rand_mask)
        random_scores.append(scorer(rand_img))

    random_mean = float(np.mean(random_scores))
    return {
        "original_score": original_score,
        "saliency_score": saliency_score,
        "saliency_drop": original_score - saliency_score,
        "random_scores": random_scores,
        "random_mean_score": random_mean,
        "random_mean_drop": original_score - random_mean,
        "random_std_score": float(np.std(random_scores)),
    }


# ============================================================
# Unified heatmap faithfulness (model-agnostic)
# ============================================================

def evaluate_faithfulness_for_heatmap(
    scorer: Callable[[torch.Tensor], float],
    image_tensor: torch.Tensor,
    heatmap: torch.Tensor,
    method_name: str,
    deletion_insertion_steps: int = 11,
    masking_fraction: float = 0.2,
    n_random: int = 10,
) -> Dict[str, Any]:
    di = evaluate_deletion_insertion(
        scorer=scorer, image_tensor=image_tensor, heatmap=heatmap,
        fractions=np.linspace(0.0, 1.0, deletion_insertion_steps),
    )
    mask_res = evaluate_masking_vs_random(
        scorer=scorer, image_tensor=image_tensor, heatmap=heatmap,
        fraction=masking_fraction, mode="zero", n_random=n_random,
    )
    blur_res = evaluate_masking_vs_random(
        scorer=scorer, image_tensor=image_tensor, heatmap=heatmap,
        fraction=masking_fraction, mode="blur", n_random=n_random,
    )

    return {
        "method": method_name,
        "deletion_auc": di["deletion_auc"],
        "insertion_auc": di["insertion_auc"],
        "saliency_mask_drop": mask_res["saliency_drop"],
        "random_mask_drop": mask_res["random_mean_drop"],
        "saliency_blur_drop": blur_res["saliency_drop"],
        "random_blur_drop": blur_res["random_mean_drop"],
        "mask_beats_random": mask_res["saliency_drop"] > mask_res["random_mean_drop"],
        "blur_beats_random": blur_res["saliency_drop"] > blur_res["random_mean_drop"],
        "deletion_result": di,
        "masking_result": mask_res,
        "blurring_result": blur_res,
    }


# ============================================================
# Multi-sample runners
# ============================================================

def run_biovilt_faithfulness(
    model, tokenizer, image_loader, matched_samples,
    heatmap_builder, method_name,
    num_samples=10, seed=42, n_random=10,
    image_loader_kwargs=None,
) -> tuple[pd.DataFrame, List[Dict]]:
    import random as _random
    rng = _random.Random(seed)
    chosen = rng.sample(list(matched_samples), min(num_samples, len(matched_samples)))

    if image_loader_kwargs is None:
        image_loader_kwargs = {}

    rows, detailed = [], []

    for sample in chosen:
        image_tensor = image_loader(sample["metadata"]["image_path"], **image_loader_kwargs)
        report_text = sample.get("metadata", {}).get("report_text", "") or ""
        scorer, _, _ = make_biovilt_scorer(model, tokenizer, report_text)

        heatmap = heatmap_builder(sample=sample, image_tensor=image_tensor, model=model, tokenizer=tokenizer)

        faith = evaluate_faithfulness_for_heatmap(
            scorer=scorer, image_tensor=image_tensor, heatmap=heatmap,
            method_name=method_name, n_random=n_random,
        )

        rows.append({
            "subject_id": sample["metadata"].get("subject_id"),
            "study_id": sample["metadata"].get("study_id"),
            "method": method_name,
            **{k: v for k, v in faith.items() if k not in ("deletion_result", "masking_result", "blurring_result")},
        })
        detailed.append({"sample": sample, "image_tensor": image_tensor, "heatmap": heatmap, "faithfulness": faith})

    return pd.DataFrame(rows), detailed


def run_cxr_clip_faithfulness(
    pipeline, test_df, cxr_clip_samples,
    label, method_name,
    heatmap_builder,
    num_samples=10, seed=42, n_random=10,
) -> tuple[pd.DataFrame, List[Dict]]:
    import random as _random
    from data_loader import CXRClipInferenceDataset
    rng = _random.Random(seed)

    pos_samples = [s for s in cxr_clip_samples if s["labels"].get(label, 0) == 1.0]
    chosen = rng.sample(pos_samples, min(num_samples, len(pos_samples)))

    scorer = make_cxr_clip_scorer(pipeline, label)
    rows, detailed = [], []

    for s in chosen:
        img_path = s["metadata"]["image_path"]
        sample_df = test_df[test_df["image_path"] == img_path].reset_index(drop=True)
        if len(sample_df) == 0:
            continue

        dataset = CXRClipInferenceDataset(sample_df, transform=pipeline["preprocess"])
        sample = dataset[0]
        image_tensor = sample["image"]

        heatmap = heatmap_builder(pipeline=pipeline, image_tensor=image_tensor, label=label)

        faith = evaluate_faithfulness_for_heatmap(
            scorer=scorer, image_tensor=image_tensor, heatmap=heatmap,
            method_name=method_name, n_random=n_random,
        )

        rows.append({
            "subject_id": s["metadata"].get("subject_id"),
            "label": label,
            "method": method_name,
            **{k: v for k, v in faith.items() if k not in ("deletion_result", "masking_result", "blurring_result")},
        })
        detailed.append({"sample": s, "image_tensor": image_tensor, "heatmap": heatmap, "faithfulness": faith})

    return pd.DataFrame(rows), detailed


def run_biovilt_integrated_gradients(
    model, tokenizer, image_loader, matched_samples,
    num_samples=10, seed=42, n_steps=50,
    image_loader_kwargs=None,
) -> tuple[pd.DataFrame, List[Dict]]:
    import random as _random
    rng = _random.Random(seed)
    chosen = rng.sample(list(matched_samples), min(num_samples, len(matched_samples)))

    if image_loader_kwargs is None:
        image_loader_kwargs = {}

    rows, results = [], []
    for sample in chosen:
        image_tensor = image_loader(sample["metadata"]["image_path"], **image_loader_kwargs)
        report_text = sample.get("metadata", {}).get("report_text", "") or ""

        ig = compute_biovilt_integrated_gradients(model, image_tensor, tokenizer, report_text, n_steps)
        rows.append({
            "subject_id": sample["metadata"].get("subject_id"),
            "study_id": sample["metadata"].get("study_id"),
            "ig_delta": float(ig["delta"].mean().item()),
        })
        results.append({"sample": sample, "image_tensor": image_tensor, **ig})

    return pd.DataFrame(rows), results


def run_cxr_clip_integrated_gradients(
    pipeline, test_df, cxr_clip_samples, label,
    num_samples=10, seed=42, n_steps=50,
) -> tuple[pd.DataFrame, List[Dict]]:
    import random as _random
    from data_loader import CXRClipInferenceDataset
    rng = _random.Random(seed)

    pos_samples = [s for s in cxr_clip_samples if s["labels"].get(label, 0) == 1.0]
    chosen = rng.sample(pos_samples, min(num_samples, len(pos_samples)))

    rows, results = [], []
    for s in chosen:
        img_path = s["metadata"]["image_path"]
        sample_df = test_df[test_df["image_path"] == img_path].reset_index(drop=True)
        if len(sample_df) == 0:
            continue

        dataset = CXRClipInferenceDataset(sample_df, transform=pipeline["preprocess"])
        sample = dataset[0]

        ig = compute_cxr_clip_integrated_gradients(pipeline, sample["image"], label, n_steps)
        rows.append({
            "subject_id": s["metadata"].get("subject_id"),
            "label": label,
            "ig_delta": float(ig["delta"].mean().item()),
        })
        results.append({"sample": s, "image_tensor": sample["image"], **ig})

    return pd.DataFrame(rows), results


# ============================================================
# Summary
# ============================================================

def summarize_faithfulness(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in df.columns if c in (
        "deletion_auc", "insertion_auc",
        "saliency_mask_drop", "random_mask_drop",
        "saliency_blur_drop", "random_blur_drop",
        "mask_beats_random", "blur_beats_random",
    )]
    return df.groupby("method", as_index=False)[metric_cols].mean()


# ============================================================
# Visualization
# ============================================================

def plot_ig_result(image_tensor, heatmap, title="Integrated Gradients",
                   alpha=0.4, cmap="magma", save_path=None):
    img = image_tensor[0].detach().cpu()
    heat = heatmap.detach().cpu()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor("white")

    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Original CXR", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(heat, cmap=cmap)
    axes[1].set_title("IG Attribution", fontsize=12)
    axes[1].axis("off")

    axes[2].imshow(img, cmap="gray")
    axes[2].imshow(heat, cmap=cmap, alpha=alpha)
    axes[2].set_title("Overlay", fontsize=12)
    axes[2].axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=250, bbox_inches="tight", facecolor="white")
    plt.show()


def plot_deletion_insertion(result, title="Deletion / Insertion", save_path=None):
    x = np.array(result["fractions"], dtype=float)
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("white")

    ax.plot(x, result["deletion_scores"], "o-", color=PALETTE["deletion"],
            linewidth=2, markersize=6, label=f"Deletion (AUC={result.get('deletion_auc', 0):.3f})")
    ax.plot(x, result["insertion_scores"], "o-", color=PALETTE["insertion"],
            linewidth=2, markersize=6, label=f"Insertion (AUC={result.get('insertion_auc', 0):.3f})")

    ax.fill_between(x, result["deletion_scores"], alpha=0.1, color=PALETTE["deletion"])
    ax.fill_between(x, result["insertion_scores"], alpha=0.1, color=PALETTE["insertion"])

    ax.set_xlabel("Fraction of top-ranked region", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(frameon=False, fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(color=PALETTE["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=250, bbox_inches="tight", facecolor="white")
    plt.show()


def plot_masking_comparison(result, title="Masking vs Random", save_path=None):
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("white")

    bars = ax.bar(
        ["Saliency-guided", "Random"],
        [result["saliency_drop"], result["random_mean_drop"]],
        color=[PALETTE["saliency"], PALETTE["random"]],
        edgecolor="white", linewidth=0.8,
    )
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Score Drop", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=250, bbox_inches="tight", facecolor="white")
    plt.show()


def plot_faithfulness_comparison_table(
    biovil_summary_df, cxr_clip_summary_df,
    title="Faithfulness Comparison: BioViL-T vs CXR-CLIP",
    save_path=None,
):
    metrics = ["deletion_auc", "insertion_auc", "saliency_mask_drop",
               "random_mask_drop", "saliency_blur_drop", "random_blur_drop",
               "mask_beats_random", "blur_beats_random"]
    metric_labels = ["Deletion AUC ↓", "Insertion AUC ↑", "Saliency Mask Drop ↑",
                     "Random Mask Drop", "Saliency Blur Drop ↑", "Random Blur Drop",
                     "Mask > Random %", "Blur > Random %"]

    rows = []
    for metric, label in zip(metrics, metric_labels):
        b_val = biovil_summary_df[metric].mean() if metric in biovil_summary_df.columns else None
        c_val = cxr_clip_summary_df[metric].mean() if metric in cxr_clip_summary_df.columns else None
        rows.append({
            "Metric": label,
            "BioViL-T": f"{b_val:.4f}" if b_val is not None else "N/A",
            "CXR-CLIP": f"{c_val:.4f}" if c_val is not None else "N/A",
        })

    df = pd.DataFrame(rows)
    n_rows, n_cols = len(rows), len(df.columns)

    fig, ax = plt.subplots(figsize=(12, 0.6 * n_rows + 2))
    fig.patch.set_facecolor("white")
    ax.axis("off")

    table = ax.table(cellText=df.values.tolist(), colLabels=df.columns.tolist(),
                     cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.7)

    for j in range(n_cols):
        table[0, j].set_facecolor(PALETTE["bg_header"])
        table[0, j].set_text_props(color="white", fontweight="bold")

    for i in range(1, n_rows + 1):
        bg = PALETTE["bg_alt"] if i % 2 == 0 else "white"
        for j in range(n_cols):
            table[i, j].set_facecolor(bg)

    ax.set_title(title, fontsize=15, fontweight="bold", pad=24)
    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=250, bbox_inches="tight", facecolor="white")
        print(f"Saved: {save_path}")
    plt.show()
    return df