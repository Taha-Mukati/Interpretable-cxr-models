# ============================================================
# evaluation/semantic_alignment.py
# ============================================================

import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from data_loader import CXRClipInferenceDataset

from interpretability.biovil_t.concept_example.tcav import (
    train_linear_cav,
    build_cav_from_samples,
    tcav_score_from_embeddings,
    similarity_tcav_score,
)
from interpretability.cxr_clip.similarity_relevance import (
    generate_similarity_relevance_map,
)


# ============================================================
# Heatmap quality metrics (shared across both models)
# ============================================================

def _normalize_heatmap(heatmap: torch.Tensor) -> torch.Tensor:
    heatmap = heatmap.float()
    heatmap = heatmap - heatmap.min()
    heatmap = heatmap / (heatmap.max() + 1e-8)
    return heatmap


def heatmap_entropy(heatmap: torch.Tensor) -> float:
    h = _normalize_heatmap(heatmap)
    p = h.reshape(-1)
    p = p / (p.sum() + 1e-8)
    entropy = -(p * (p + 1e-8).log()).sum().item()
    return float(entropy)


def peak_localisation_score(heatmap: torch.Tensor) -> float:
    h = _normalize_heatmap(heatmap)
    return float(h.max().item() / (h.mean().item() + 1e-8))


def heatmap_sparsity(heatmap: torch.Tensor, threshold: float = 0.7) -> float:
    h = _normalize_heatmap(heatmap)
    return float((h >= threshold).float().mean().item())


# ============================================================
# 1. CXR-CLIP sample conversion
# ============================================================

def build_cxr_clip_samples(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    targets: List[str],
) -> List[Dict[str, Any]]:
    samples = []
    for i in range(len(metadata)):
        row = metadata.iloc[i]
        labels = {}
        for t in targets:
            col = f"{t}_true"
            if col in row.index:
                labels[t] = float(row[col])

        samples.append({
            "image_emb": torch.tensor(embeddings[i], dtype=torch.float32),
            "metadata": {
                "image_path": row.get("image_path"),
                "study_id": row.get("study_id"),
                "subject_id": row.get("subject_id"),
                "dicom_id": row.get("dicom_id"),
            },
            "labels": labels,
        })
    return samples


def select_samples_by_label(
    samples: List[Dict],
    concept_name: str,
    positive_label: float = 1.0,
    max_positive: Optional[int] = None,
    max_negative: Optional[int] = None,
    seed: int = 42,
):
    rng = random.Random(seed)
    positives, negatives = [], []

    for s in samples:
        labels = s.get("labels")
        if labels is None or concept_name not in labels:
            continue
        if float(labels[concept_name]) == positive_label:
            positives.append(s)
        else:
            negatives.append(s)

    if max_positive and len(positives) > max_positive:
        positives = rng.sample(positives, max_positive)
    if max_negative and len(negatives) > max_negative:
        negatives = rng.sample(negatives, max_negative)

    return positives, negatives


# ============================================================
# CXR-CLIP heatmap quality evaluation
# ============================================================

def evaluate_cxr_clip_heatmap_quality(
    cxr_clip_samples,
    test_df,
    pipeline,
    pathologies,
    max_samples=200,
    seed=42,
):
    rng = random.Random(seed)
    rows = []

    for pathology in pathologies:
        pos_samples = [s for s in cxr_clip_samples
                       if s["labels"].get(pathology, 0) == 1.0]

        selected = rng.sample(pos_samples, min(max_samples, len(pos_samples)))

        for s in selected:
            img_path = s["metadata"]["image_path"]
            sample_df = test_df[test_df["image_path"] == img_path].reset_index(drop=True)

            if len(sample_df) == 0:
                continue

            sample_dataset = CXRClipInferenceDataset(sample_df, transform=pipeline["preprocess"])
            sample = sample_dataset[0]

            hmap = generate_similarity_relevance_map(
                pipeline=pipeline,
                image_tensor=sample["image"],
                label=pathology,
                use_margin=True,
            )

            heat_t = torch.tensor(hmap, dtype=torch.float32)

            rows.append({
                "subject_id": s["metadata"].get("subject_id"),
                "token_query": pathology.lower(),
                "pathology": pathology,
                "entropy": heatmap_entropy(heat_t),
                "peak_score": peak_localisation_score(heat_t),
                "sparsity": heatmap_sparsity(heat_t),
            })

    return pd.DataFrame(rows)


# ============================================================
# 2. Unified TCAV builder
# ============================================================

def build_tcav_concepts(
    samples: List[Dict],
    pathologies: List[str],
    embedding_key: str = "image_emb",
    max_positive: int = 100,
    max_negative: int = 100,
    seed: int = 42,
    model_name: str = "",
    has_text_emb: bool = True,
) -> Dict[str, Dict]:
    concepts = {}

    for pathology in pathologies:
        print(f"\n[{model_name}] Building TCAV: {pathology}")

        positives, negatives = select_samples_by_label(
            samples=samples,
            concept_name=pathology,
            max_positive=max_positive,
            max_negative=max_negative,
            seed=seed,
        )

        if len(positives) == 0 or len(negatives) == 0:
            print(f"  Skipping {pathology}: insufficient samples.")
            continue

        cav = build_cav_from_samples(
            positives=positives,
            negatives=negatives,
            embedding_key=embedding_key,
        )

        tcav_emb, directional = tcav_score_from_embeddings(
            samples=positives + negatives,
            cav=cav,
            embedding_key=embedding_key,
        )

        tcav_sim, sim_directional = None, None
        if has_text_emb:
            try:
                tcav_sim, sim_directional = similarity_tcav_score(
                    samples=positives + negatives,
                    cav=cav,
                    embedding_key=embedding_key,
                )
            except Exception as e:
                print(f"  Similarity TCAV failed: {e}")

        concepts[pathology] = {
            "positives": positives,
            "negatives": negatives,
            "cav": cav,
            "tcav_emb": tcav_emb,
            "tcav_sim": tcav_sim,
            "directional": directional,
            "sim_directional": sim_directional,
        }

        print(f"  Positives: {len(positives)}, Negatives: {len(negatives)}")
        print(f"  Embedding TCAV: {tcav_emb:.4f}")
        if tcav_sim is not None:
            print(f"  Similarity TCAV: {tcav_sim:.4f}")

    return concepts


# ============================================================
# 3. Balanced TCAV comparison
# ============================================================

def build_balanced_tcav_comparison(
    biovil_samples: List[Dict],
    cxr_clip_samples: List[Dict],
    pathologies: List[str],
    biovil_embedding_key: str = "image_emb",
    cxr_clip_embedding_key: str = "image_emb",
    max_per_class: int = 100,
    seed: int = 42,
) -> Dict[str, Dict]:
    comparison = {}

    for pathology in pathologies:
        b_pos, b_neg = select_samples_by_label(
            biovil_samples, pathology, max_positive=None, max_negative=None, seed=seed
        )
        c_pos, c_neg = select_samples_by_label(
            cxr_clip_samples, pathology, max_positive=None, max_negative=None, seed=seed
        )

        n_pos = min(len(b_pos), len(c_pos), max_per_class)
        n_neg = min(len(b_neg), len(c_neg), max_per_class)

        if n_pos == 0 or n_neg == 0:
            print(f"Skipping {pathology}: insufficient samples in one or both models.")
            continue

        rng = random.Random(seed)
        b_pos_sel = rng.sample(b_pos, n_pos)
        b_neg_sel = rng.sample(b_neg, n_neg)

        rng = random.Random(seed)
        c_pos_sel = rng.sample(c_pos, n_pos)
        c_neg_sel = rng.sample(c_neg, n_neg)

        print(f"\n{pathology}: {n_pos} pos, {n_neg} neg per model")

        b_cav = build_cav_from_samples(b_pos_sel, b_neg_sel, embedding_key=biovil_embedding_key)
        b_tcav, b_dir = tcav_score_from_embeddings(
            b_pos_sel + b_neg_sel, b_cav, embedding_key=biovil_embedding_key
        )
        b_sim, b_sim_dir = similarity_tcav_score(
            b_pos_sel + b_neg_sel, b_cav, embedding_key=biovil_embedding_key
        )

        c_cav = build_cav_from_samples(c_pos_sel, c_neg_sel, embedding_key=cxr_clip_embedding_key)
        c_tcav, c_dir = tcav_score_from_embeddings(
            c_pos_sel + c_neg_sel, c_cav, embedding_key=cxr_clip_embedding_key
        )

        c_sim = None
        if c_pos_sel[0].get("text_emb") is not None:
            c_sim, _ = similarity_tcav_score(
                c_pos_sel + c_neg_sel, c_cav, embedding_key=cxr_clip_embedding_key
            )

        comparison[pathology] = {
            "n_pos": n_pos,
            "n_neg": n_neg,
            "biovil_t": {
                "cav": b_cav, "tcav_emb": b_tcav, "tcav_sim": b_sim,
                "directional": b_dir, "positives": b_pos_sel, "negatives": b_neg_sel,
            },
            "cxr_clip": {
                "cav": c_cav, "tcav_emb": c_tcav, "tcav_sim": c_sim,
                "directional": c_dir, "positives": c_pos_sel, "negatives": c_neg_sel,
            },
        }

        print(f"  BioViL-T  emb={b_tcav:.4f}  sim={b_sim:.4f}")
        print(f"  CXR-CLIP  emb={c_tcav:.4f}  sim={c_sim if c_sim else 'N/A'}")

    return comparison


# ============================================================
# 4. Summary tables
# ============================================================

def tcav_comparison_to_dataframe(comparison: Dict) -> pd.DataFrame:
    rows = []
    for pathology, data in comparison.items():
        rows.append({
            "Pathology": pathology,
            "Model": "BioViL-T",
            "N_pos": data["n_pos"],
            "N_neg": data["n_neg"],
            "Embedding_TCAV": data["biovil_t"]["tcav_emb"],
            "Similarity_TCAV": data["biovil_t"]["tcav_sim"],
        })
        rows.append({
            "Pathology": pathology,
            "Model": "CXR-CLIP",
            "N_pos": data["n_pos"],
            "N_neg": data["n_neg"],
            "Embedding_TCAV": data["cxr_clip"]["tcav_emb"],
            "Similarity_TCAV": data["cxr_clip"]["tcav_sim"],
        })
    return pd.DataFrame(rows)


def tcav_single_model_to_dataframe(concepts: Dict, model_name: str = "") -> pd.DataFrame:
    rows = []
    for pathology, data in concepts.items():
        rows.append({
            "Pathology": pathology,
            "Model": model_name,
            "N_pos": len(data["positives"]),
            "N_neg": len(data["negatives"]),
            "Embedding_TCAV": data["tcav_emb"],
            "Similarity_TCAV": data["tcav_sim"],
        })
    return pd.DataFrame(rows)


# ============================================================
# 5. Colour palette
# ============================================================

PALETTE = {
    "BioViL-T":            "#0077B6",  # deep ocean blue
    "CXR-CLIP":            "#E63946",  # vivid coral red
    "BioViL-T (Temporal)": "#2A9D8F",  # teal green
    "accent_purple":       "#7209B7",
    "accent_amber":        "#F4A261",
    "accent_emerald":      "#06D6A0",
    "positive":            "#0077B6",
    "negative":            "#E63946",
    "grid":                "#E0E0E0",
    "bg_header":           "#1B2838",
    "bg_alt_row":          "#F0F4F8",
    "bg_overall_row":      "#D4EDDA",
    "baseline":            "#ADB5BD",
}


# ============================================================
# 6. Visualization
# ============================================================

def plot_tcav_comparison_bar(
    comparison_df: pd.DataFrame,
    metric: str = "Embedding_TCAV",
    title: str = "TCAV comparison across all methods",
    save_path: Optional[str] = None,
    figsize: tuple = (12, 6),
):
    pathologies = comparison_df["Pathology"].unique()
    models = comparison_df["Model"].unique()

    x = np.arange(len(pathologies))
    width = 0.22
    n_models = len(models)

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")

    for i, model in enumerate(models):
        subset = comparison_df[comparison_df["Model"] == model]
        scores = []
        for p in pathologies:
            row = subset[subset["Pathology"] == p]
            scores.append(row[metric].values[0] if len(row) > 0 else 0)

        offset = (i - (n_models - 1) / 2) * width
        color = PALETTE.get(model, f"C{i}")

        bars = ax.bar(
            x + offset, scores, width,
            label=model, color=color,
            edgecolor="white", linewidth=1,
            alpha=0.85, zorder=3,
        )

        for bar, val in zip(bars, scores):
            ax.annotate(
                f"{val:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=10, fontweight="bold",
                color="#1a1a1a",
                bbox=dict(
                    boxstyle="round,pad=0.15",
                    facecolor="white",
                    edgecolor=color,
                    linewidth=0.8,
                    alpha=0.9,
                ),
            )

    ax.axhline(
        0.5, linestyle="--", linewidth=1.2,
        color=PALETTE["baseline"], alpha=0.6,
        label="Random baseline (0.5)", zorder=2,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(pathologies, fontsize=12, fontweight="bold")
    ax.set_ylabel("TCAV score", fontsize=13)
    ax.set_title(title, fontsize=15, fontweight="bold", pad=16)
    ax.set_ylim(0, 1.15)
    ax.legend(
        fontsize=10, frameon=True, fancybox=True,
        framealpha=0.9, edgecolor=PALETTE["grid"],
        loc="upper left",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=11)

    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=250, bbox_inches="tight", facecolor="white")
        print(f"Saved: {save_path}")
    plt.show()
    return fig


def plot_tcav_table_figure(
    df: pd.DataFrame,
    title: str = "TCAV Results",
    save_path: Optional[str] = None,
    figsize: Optional[tuple] = None,
):
    display_df = df.copy()
    for col in ["Embedding_TCAV", "Similarity_TCAV"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(
                lambda x: f"{x:.4f}" if x is not None and not (isinstance(x, float) and np.isnan(x)) else "N/A"
            )

    cols = display_df.columns.tolist()
    data = display_df.values.tolist()

    if figsize is None:
        figsize = (2.8 * len(cols), 0.55 * len(data) + 1.8)

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    ax.axis("off")

    table = ax.table(cellText=data, colLabels=cols, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.7)

    for j in range(len(cols)):
        table[0, j].set_facecolor(PALETTE["bg_header"])
        table[0, j].set_text_props(color="white", fontweight="bold", fontsize=11)

    for i in range(1, len(data) + 1):
        bg = PALETTE["bg_alt_row"] if i % 2 == 0 else "white"
        for j in range(len(cols)):
            table[i, j].set_facecolor(bg)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=22)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=250, bbox_inches="tight", facecolor="white")
        print(f"Saved: {save_path}")
    plt.show()
    return fig


def plot_semantic_alignment_heatmap_comparison(
    biovil_metrics_df: pd.DataFrame,
    cxr_clip_metrics_df: pd.DataFrame,
    title: str = "Semantic alignment: heatmap quality comparison",
    save_path: Optional[str] = None,
    figsize: tuple = (16, 5),
):
    metrics = ["entropy", "peak_score", "sparsity"]
    metric_labels = [
        "Entropy\n(lower → more focused)",
        "Peak score\n(higher → stronger peak)",
        "Sparsity\n(lower → more selective)",
    ]

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.patch.set_facecolor("white")

    box_colors = [PALETTE["BioViL-T"], PALETTE["CXR-CLIP"]]

    for ax, metric, label in zip(axes, metrics, metric_labels):
        b_vals = biovil_metrics_df[metric].dropna().values
        c_vals = cxr_clip_metrics_df[metric].dropna().values

        bp = ax.boxplot(
            [b_vals, c_vals],
            positions=[1, 2],
            patch_artist=True,
            widths=0.45,
            medianprops=dict(color="white", linewidth=2),
            whiskerprops=dict(color="#555555", linewidth=1.2),
            capprops=dict(color="#555555", linewidth=1.2),
            flierprops=dict(marker="o", markersize=4, alpha=0.4, markeredgecolor="#888888"),
            boxprops=dict(linewidth=1.2),
        )

        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor(color)

        # Annotate medians with black text, white background for readability
        for i, (vals, pos) in enumerate(zip([b_vals, c_vals], [1, 2])):
            med = np.median(vals)
            ax.annotate(
                f"{med:.2f}",
                xy=(pos, med),
                xytext=(pos + 0.35, med),
                fontsize=11,
                fontweight="bold",
                color="#1a1a1a",
                va="center",
                ha="left",
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="white",
                    edgecolor=box_colors[i],
                    linewidth=1,
                    alpha=0.9,
                ),
            )

        ax.set_title(label, fontsize=13, fontweight="bold", pad=12)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["BioViL-T", "CXR-CLIP"], fontsize=12)
        ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.6, alpha=0.7)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.5)
        ax.spines["bottom"].set_linewidth(0.5)
        ax.tick_params(axis="y", labelsize=11)

    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.04)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=250, bbox_inches="tight", facecolor="white")
        print(f"Saved: {save_path}")
    plt.show()
    return fig


def build_semantic_alignment_summary(
    biovil_alignment_df,
    cxr_clip_alignment_df,
    comparison_df,
    save_path=None,
):
    b_means = biovil_alignment_df[["entropy", "peak_score", "sparsity"]].mean()
    c_means = cxr_clip_alignment_df[["entropy", "peak_score", "sparsity"]].mean()

    b_tcav = comparison_df[comparison_df["Model"] == "BioViL-T"]["Embedding_TCAV"].mean()
    c_tcav = comparison_df[comparison_df["Model"] == "CXR-CLIP"]["Embedding_TCAV"].mean()
    b_sim = comparison_df[comparison_df["Model"] == "BioViL-T"]["Similarity_TCAV"].mean()

    rows = [
        {"Metric": "Embedding TCAV (mean)",
         "BioViL-T": f"{b_tcav:.4f}", "CXR-CLIP": f"{c_tcav:.4f}",
         "Notes": "Higher = more concept-sensitive"},
        {"Metric": "Similarity TCAV (mean)",
         "BioViL-T": f"{b_sim:.4f}" if b_sim is not None and not np.isnan(b_sim) else "N/A",
         "CXR-CLIP": "N/A",
         "Notes": "BioViL-T only (paired reports)"},
        {"Metric": "Heatmap Entropy (mean)",
         "BioViL-T": f"{b_means['entropy']:.4f}", "CXR-CLIP": f"{c_means['entropy']:.4f}",
         "Notes": "Lower = more focused"},
        {"Metric": "Peak Score (mean)",
         "BioViL-T": f"{b_means['peak_score']:.4f}", "CXR-CLIP": f"{c_means['peak_score']:.4f}",
         "Notes": "Higher = stronger localisation"},
        {"Metric": "Sparsity (mean)",
         "BioViL-T": f"{b_means['sparsity']:.4f}", "CXR-CLIP": f"{c_means['sparsity']:.4f}",
         "Notes": "Lower = more selective"},
        {"Metric": "Cross-Modal Attention",
         "BioViL-T": "✓ Supported", "CXR-CLIP": "✗ Not available",
         "Notes": "Token-to-patch alignment"},
        {"Metric": "Similarity Relevance",
         "BioViL-T": "✓ Supported", "CXR-CLIP": "✓ Supported",
         "Notes": "Patch-to-text similarity map"},
        {"Metric": "Prototype Retrieval",
         "BioViL-T": "✓ Supported", "CXR-CLIP": "✓ Supported",
         "Notes": "Embedding nearest neighbours"},
        {"Metric": "Report Retrieval",
         "BioViL-T": "✓ Supported", "CXR-CLIP": "✗ Not available",
         "Notes": "Requires paired report embeddings"},
    ]

    summary_df = pd.DataFrame(rows)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        n_rows = len(rows)
        n_cols = len(summary_df.columns)
        fig, ax = plt.subplots(figsize=(16, 0.6 * n_rows + 2))
        fig.patch.set_facecolor("white")
        ax.axis("off")

        table = ax.table(
            cellText=summary_df.values.tolist(),
            colLabels=summary_df.columns.tolist(),
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 1.7)

        for j in range(n_cols):
            table[0, j].set_facecolor(PALETTE["bg_header"])
            table[0, j].set_text_props(color="white", fontweight="bold", fontsize=11)

        for i in range(1, n_rows + 1):
            bg = PALETTE["bg_alt_row"] if i % 2 == 0 else "white"
            for j in range(n_cols):
                cell = table[i, j]
                cell.set_facecolor(bg)
                text = cell.get_text().get_text()
                if "✓" in text:
                    cell.get_text().set_color(PALETTE["BioViL-T (Temporal)"])
                    cell.get_text().set_fontweight("bold")
                elif "✗" in text:
                    cell.get_text().set_color(PALETTE["CXR-CLIP"])
                    cell.get_text().set_fontweight("bold")

        ax.set_title("Semantic Alignment: Complete Comparison",
                      fontsize=15, fontweight="bold", pad=24)
        plt.tight_layout()
        fig.savefig(save_path, dpi=250, bbox_inches="tight", facecolor="white")
        print(f"Saved: {save_path}")
        plt.show()

    return summary_df