# ============================================================
# evaluation/clinical_plausibility.py
# ============================================================

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image


# ============================================================
# Colour palette
# ============================================================

PALETTE = {
    "pass": "#2A9D8F",
    "fail": "#E63946",
    "warning": "#F4A261",
    "BioViL-T": "#0077B6",
    "CXR-CLIP": "#E63946",
    "grid": "#E0E0E0",
    "bg_header": "#1B2838",
    "bg_alt": "#F0F4F8",
}


# ============================================================
# 1. Artefact-check heuristics
# ============================================================

def check_border_focus(
    heatmap: torch.Tensor,
    border_fraction: float = 0.1,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Check if the heatmap peak activation is concentrated in image borders,
    which would suggest artefact reliance rather than clinical reasoning.
    
    Returns dict with border activation ratio and pass/fail flag.
    """
    h, w = heatmap.shape
    heat = heatmap.float()
    heat = heat - heat.min()
    heat = heat / (heat.max() + 1e-8)

    border_h = max(1, int(h * border_fraction))
    border_w = max(1, int(w * border_fraction))

    border_mask = torch.zeros_like(heat)
    border_mask[:border_h, :] = 1.0      # top
    border_mask[-border_h:, :] = 1.0     # bottom
    border_mask[:, :border_w] = 1.0      # left
    border_mask[:, -border_w:] = 1.0     # right

    border_activation = (heat * border_mask).sum().item()
    total_activation = heat.sum().item() + 1e-8
    border_ratio = border_activation / total_activation

    # Check if peak is in border
    peak_idx = heat.argmax().item()
    peak_row, peak_col = divmod(peak_idx, w)
    peak_in_border = (
        peak_row < border_h or peak_row >= h - border_h or
        peak_col < border_w or peak_col >= w - border_w
    )

    return {
        "border_activation_ratio": border_ratio,
        "peak_in_border": peak_in_border,
        "pass": border_ratio < threshold and not peak_in_border,
        "check": "border_focus",
    }


def check_rib_overfocus(
    heatmap: torch.Tensor,
    horizontal_stripe_fraction: float = 0.05,
    max_stripe_concentration: float = 0.4,
) -> Dict[str, Any]:
    """
    Check if activation is concentrated in narrow horizontal stripes,
    which could indicate rib/edge artefact reliance.
    
    Ribs appear as roughly horizontal structures, so a model fixating
    on them would show high activation in thin horizontal bands.
    """
    h, w = heatmap.shape
    heat = heatmap.float()
    heat = heat - heat.min()
    heat = heat / (heat.max() + 1e-8)

    # Sum activation per row
    row_sums = heat.sum(dim=1)  # [H]
    row_sums = row_sums / (row_sums.sum() + 1e-8)

    # Check if top-k rows contain too much activation
    k = max(1, int(h * horizontal_stripe_fraction))
    top_row_activation = row_sums.topk(k).values.sum().item()

    return {
        "top_stripe_activation": top_row_activation,
        "pass": top_row_activation < max_stripe_concentration,
        "check": "rib_overfocus",
    }


def check_centre_bias(
    heatmap: torch.Tensor,
    centre_fraction: float = 0.25,
    min_centre_ratio: float = 0.15,
    max_centre_ratio: float = 0.85,
) -> Dict[str, Any]:
    """
    Check if activation is excessively concentrated in the centre
    (possible lazy shortcut) or completely avoids it.
    """
    h, w = heatmap.shape
    heat = heatmap.float()
    heat = heat - heat.min()
    heat = heat / (heat.max() + 1e-8)

    ch = int(h * centre_fraction)
    cw = int(w * centre_fraction)
    r0 = (h - ch) // 2
    c0 = (w - cw) // 2

    centre_activation = heat[r0:r0+ch, c0:c0+cw].sum().item()
    total_activation = heat.sum().item() + 1e-8
    centre_ratio = centre_activation / total_activation

    return {
        "centre_activation_ratio": centre_ratio,
        "pass": min_centre_ratio < centre_ratio < max_centre_ratio,
        "check": "centre_bias",
    }


def check_activation_spread(
    heatmap: torch.Tensor,
    high_threshold: float = 0.7,
    min_spread: float = 0.01,
    max_spread: float = 0.5,
) -> Dict[str, Any]:
    """
    Check that high-activation region is neither a single pixel
    nor the entire image.
    """
    heat = heatmap.float()
    heat = heat - heat.min()
    heat = heat / (heat.max() + 1e-8)

    spread = (heat >= high_threshold).float().mean().item()

    return {
        "high_activation_spread": spread,
        "pass": min_spread < spread < max_spread,
        "check": "activation_spread",
    }


def run_all_artefact_checks(heatmap: torch.Tensor) -> Dict[str, Dict]:
    """Run all artefact heuristics on one heatmap."""
    return {
        "border_focus": check_border_focus(heatmap),
        "rib_overfocus": check_rib_overfocus(heatmap),
        "centre_bias": check_centre_bias(heatmap),
        "activation_spread": check_activation_spread(heatmap),
    }


# ============================================================
# 2. Batch artefact evaluation
# ============================================================

def evaluate_artefact_checks_batch(
    heatmaps: List[torch.Tensor],
    sample_ids: List[str],
    method_name: str = "",
) -> pd.DataFrame:
    """Run artefact checks on a batch of heatmaps."""
    rows = []

    for hmap, sid in zip(heatmaps, sample_ids):
        checks = run_all_artefact_checks(hmap)
        row = {"sample_id": sid, "method": method_name}

        for check_name, result in checks.items():
            row[f"{check_name}_pass"] = result["pass"]
            for k, v in result.items():
                if k not in ("pass", "check"):
                    row[f"{check_name}_{k}"] = v

        rows.append(row)

    return pd.DataFrame(rows)


def summarize_artefact_checks(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize pass rates across all samples."""
    pass_cols = [c for c in df.columns if c.endswith("_pass")]
    summary_rows = []

    for col in pass_cols:
        check_name = col.replace("_pass", "")
        pass_rate = df[col].mean()
        n_pass = int(df[col].sum())
        n_total = len(df)
        summary_rows.append({
            "Check": check_name,
            "Pass Rate": f"{pass_rate:.1%}",
            "Passed": n_pass,
            "Total": n_total,
        })

    return pd.DataFrame(summary_rows)


# ============================================================
# 3. Qualitative inspection panels
# ============================================================

def plot_clinical_plausibility_panel(
    image_path: str,
    heatmap: torch.Tensor,
    artefact_results: Dict[str, Dict],
    method_name: str = "",
    pathology: str = "",
    ground_truth: List[str] = None,
    alpha: float = 0.4,
    cmap: str = "magma",
    save_path: Optional[str] = None,
):
    """
    Clinical plausibility inspection panel:
    Original | Heatmap overlay | Artefact check results
    """
    img = Image.open(image_path).convert("L")
    img_arr = np.array(img).astype(np.float32) / 255.0
    h, w = img_arr.shape

    heat = heatmap.detach().cpu().float()
    if heat.shape != (h, w):
        heat = F.interpolate(
            heat.unsqueeze(0).unsqueeze(0), size=(h, w),
            mode="bilinear", align_corners=False,
        )[0, 0]
    heat = heat - heat.min()
    heat = heat / (heat.max() + 1e-8)

    # Find peak
    peak_idx = heat.argmax().item()
    peak_y, peak_x = divmod(peak_idx, w)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor("white")

    # Original
    axes[0].imshow(img_arr, cmap="gray")
    axes[0].set_title("Original CXR", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    # Overlay with peak
    axes[1].imshow(img_arr, cmap="gray")
    axes[1].imshow(heat.numpy(), cmap=cmap, alpha=alpha)
    axes[1].plot(peak_x, peak_y, "x", color="cyan", markersize=14, markeredgewidth=2)
    axes[1].set_title(f"Heatmap + Peak\n{method_name}", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    # Artefact check results
    axes[2].axis("off")
    check_text = f"Artefact Check Results\n{'=' * 35}\n\n"

    for check_name, result in artefact_results.items():
        status = "PASS" if result["pass"] else "FAIL"
        color = PALETTE["pass"] if result["pass"] else PALETTE["fail"]
        check_text += f"{check_name}: {status}\n"

        for k, v in result.items():
            if k not in ("pass", "check"):
                if isinstance(v, float):
                    check_text += f"  {k}: {v:.4f}\n"
                else:
                    check_text += f"  {k}: {v}\n"
        check_text += "\n"

    all_pass = all(r["pass"] for r in artefact_results.values())
    overall = "ALL CHECKS PASSED" if all_pass else "SOME CHECKS FAILED"
    overall_color = PALETTE["pass"] if all_pass else PALETTE["fail"]

    axes[2].text(
        0.05, 0.95, check_text, transform=axes[2].transAxes,
        fontsize=10, family="monospace", va="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor=PALETTE["grid"]),
    )
    axes[2].text(
        0.05, 0.02, overall, transform=axes[2].transAxes,
        fontsize=13, fontweight="bold", color=overall_color, va="bottom",
    )

    gt_text = ", ".join(ground_truth) if ground_truth else "N/A"
    fig.suptitle(
        f"Clinical Plausibility | {method_name} | {pathology}\nGround Truth: {gt_text}",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=250, bbox_inches="tight", facecolor="white")
        print(f"Saved: {save_path}")

    plt.show()
    plt.close(fig)


def plot_artefact_summary_table(
    summary_df: pd.DataFrame,
    title: str = "Artefact Check Summary",
    save_path: Optional[str] = None,
):
    """Render artefact check summary as a table figure."""
    n_rows = len(summary_df)
    n_cols = len(summary_df.columns)

    fig, ax = plt.subplots(figsize=(10, 0.6 * n_rows + 2))
    fig.patch.set_facecolor("white")
    ax.axis("off")

    table = ax.table(
        cellText=summary_df.values.tolist(),
        colLabels=summary_df.columns.tolist(),
        cellLoc="center", loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.7)

    for j in range(n_cols):
        table[0, j].set_facecolor(PALETTE["bg_header"])
        table[0, j].set_text_props(color="white", fontweight="bold")

    for i in range(1, n_rows + 1):
        bg = PALETTE["bg_alt"] if i % 2 == 0 else "white"
        for j in range(n_cols):
            cell = table[i, j]
            cell.set_facecolor(bg)
            text = cell.get_text().get_text()
            if "100.0%" in text:
                cell.get_text().set_color(PALETTE["pass"])
                cell.get_text().set_fontweight("bold")
            elif "0.0%" in text:
                cell.get_text().set_color(PALETTE["fail"])
                cell.get_text().set_fontweight("bold")

    ax.set_title(title, fontsize=15, fontweight="bold", pad=24)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=250, bbox_inches="tight", facecolor="white")
        print(f"Saved: {save_path}")

    plt.show()
    return fig


def plot_artefact_comparison(
    biovil_summary: pd.DataFrame,
    cxr_clip_summary: pd.DataFrame,
    title: str = "Artefact Check Comparison",
    save_path: Optional[str] = None,
):
    """Side-by-side bar chart comparing pass rates."""
    checks = biovil_summary["Check"].tolist()

    b_rates = [float(r.replace("%", "")) / 100 for r in biovil_summary["Pass Rate"]]
    c_rates = [float(r.replace("%", "")) / 100 for r in cxr_clip_summary["Pass Rate"]]

    x = np.arange(len(checks))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("white")

    bars1 = ax.bar(x - width/2, b_rates, width, label="BioViL-T",
                   color=PALETTE["BioViL-T"], edgecolor="white", linewidth=0.8)
    bars2 = ax.bar(x + width/2, c_rates, width, label="CXR-CLIP",
                   color=PALETTE["CXR-CLIP"], edgecolor="white", linewidth=0.8)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                    f"{h:.0%}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(checks, fontsize=11, rotation=15)
    ax.set_ylabel("Pass Rate", fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(frameon=False, fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.6)
    ax.set_axisbelow(True)

    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=250, bbox_inches="tight", facecolor="white")
        print(f"Saved: {save_path}")
    plt.show()
    return fig