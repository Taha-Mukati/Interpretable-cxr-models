from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from data_loader import CXRClipInferenceDataset

from interpretability.cxr_clip.grad_cam import (
    generate_gradcam,
    generate_gradcam_plus_plus,
)
from interpretability.cxr_clip.similarity_relevance import (
    generate_similarity_relevance_map,
)


SUPPORTED_METHODS = ("similarity", "gradcam", "gradcam++")






# =========================================================
# Basic heatmap utilities
# =========================================================

def normalize_heatmap(heatmap: torch.Tensor) -> torch.Tensor:
    """
    Normalize a 2D heatmap to [0, 1].
    """
    heatmap = heatmap.float()
    heatmap = heatmap - heatmap.min()
    heatmap = heatmap / (heatmap.max() + 1e-8)
    return heatmap


def ensure_same_size(
    heatmap: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Resize heatmap to the mask size if needed.

    Args:
        heatmap: [H, W]
        mask:    [H, W]

    Returns:
        resized, normalized heatmap and binary mask
    """
    heatmap = heatmap.float()
    mask = mask.float()

    if heatmap.shape != mask.shape:
        heatmap = F.interpolate(
            heatmap.unsqueeze(0).unsqueeze(0),
            size=mask.shape,
            mode="bilinear",
            align_corners=False,
        )[0, 0]

    heatmap = normalize_heatmap(heatmap)
    mask = (mask > 0).float()
    return heatmap, mask


def get_peak_point(heatmap: torch.Tensor) -> tuple[int, int]:
    """
    Return (row, col) of the peak activation point.
    """
    flat_idx = int(heatmap.reshape(-1).argmax().item())
    h, w = heatmap.shape
    row = flat_idx // w
    col = flat_idx % w
    return row, col


# =========================================================
# CheXmask loading
# =========================================================

def collect_needed_dicom_ids(samples, include_prior=False):
    ids = set()

    for s in samples:
        meta = s.get("metadata", {})

        candidates = [
            s.get("dicom_id"),
            meta.get("dicom_id"),
            meta.get("current_dicom_id"),
        ]

        if include_prior:
            candidates.append(meta.get("prior_dicom_id"))

        for dicom_id in candidates:
            if dicom_id is not None:
                ids.add(str(dicom_id).strip())

    return ids


def load_chexmask_mimic_csv(
    csv_path,
    needed_dicom_ids,
    min_dice_mean=0.7,
    chunksize=50000,
):
    required_cols = [
        "dicom_id",
        "Dice RCA (Mean)",
        "Left Lung",
        "Right Lung",
        "Heart",
        "Height",
        "Width",
    ]

    rows = []

    needed_dicom_ids = {str(x) for x in needed_dicom_ids}

    for chunk in pd.read_csv(
        csv_path,
        usecols=required_cols,
        chunksize=chunksize,
        dtype={
            "dicom_id": "string",
            "Dice RCA (Mean)": "float32",
            "Left Lung": "string",
            "Right Lung": "string",
            "Heart": "string",
            "Height": "int32",
            "Width": "int32",
        },
    ):
        chunk = chunk[chunk["dicom_id"].isin(needed_dicom_ids)]
        chunk = chunk[chunk["Dice RCA (Mean)"] >= min_dice_mean]

        if not chunk.empty:
            rows.append(chunk)

    if not rows:
        return pd.DataFrame(columns=required_cols)

    return pd.concat(rows, ignore_index=True)


def decode_rle(rle_string, height, width):
    if pd.isna(rle_string) or str(rle_string).strip() == "":
        return torch.zeros((height, width), dtype=torch.uint8)

    s = list(map(int, str(rle_string).split()))
    starts = np.asarray(s[0::2], dtype=np.int64) - 1
    lengths = np.asarray(s[1::2], dtype=np.int64)
    ends = starts + lengths

    flat = np.zeros(height * width, dtype=np.uint8)
    for start, end in zip(starts, ends):
        flat[start:end] = 1

    mask = flat.reshape((height, width))
    return torch.from_numpy(mask)


def build_chexmask_lookup(df):
    """
    lookup[dicom_id] = {
        "heart": Tensor[H, W],
        "left_lung": Tensor[H, W],
        "right_lung": Tensor[H, W],
        "height": H,
        "width": W,
        "dice_rca_mean": float,
    }
    """
    lookup = {}

    for _, row in df.iterrows():
        dicom_id = str(row["dicom_id"]).strip()
        h = int(row["Height"])
        w = int(row["Width"])

        lookup[dicom_id] = {
            "heart": decode_rle(row["Heart"], h, w),
            "left_lung": decode_rle(row["Left Lung"], h, w),
            "right_lung": decode_rle(row["Right Lung"], h, w),
            "height": h,
            "width": w,
            "dice_rca_mean": float(row["Dice RCA (Mean)"]),
        }

    return lookup


# =========================================================
# MIMIC <-> CheXmask matching
# =========================================================

PATHOLOGY_TO_ANATOMY = {
    "Cardiomegaly": ["heart"],
    "Pleural Effusion": ["left_lung", "right_lung"],
    "Pneumothorax": ["left_lung", "right_lung"],
}


def build_pathology_mask(chexmask_masks, pathology_name, pathology_to_anatomy=None):
    if pathology_to_anatomy is None:
        pathology_to_anatomy = PATHOLOGY_TO_ANATOMY

    anatomy_keys = pathology_to_anatomy.get(pathology_name)
    if anatomy_keys is None:
        return None

    combined = None
    for key in anatomy_keys:
        mask = chexmask_masks.get(key)
        if mask is None:
            continue

        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(np.array(mask), dtype=torch.float32)

        if combined is None:
            combined = mask.clone()
        else:
            combined = torch.clamp(combined + mask, 0, 1)

    return combined

def mimic_image_id_from_path(image_path: str | Path) -> str:
    """
    Extract CheXmask-compatible image ID from a MIMIC-CXR-JPG path.

    Example:
        /.../p10/p10000032/s50414267/02aa804e-bde0afdd.jpg
        -> 02aa804e-bde0afdd
    """
    return Path(image_path).stem


def attach_chexmask_to_samples(samples, chexmask_lookup):
    matched = []
    missing = []

    for s in samples:
        meta = s["metadata"]
        dicom_id = meta.get("dicom_id") or meta.get("current_dicom_id")

        if dicom_id is None:
            missing.append(("NO_DICOM_ID", s))
            continue

        dicom_id = str(dicom_id).strip()

        if dicom_id in chexmask_lookup:
            s_new = dict(s)
            s_new["chexmask_masks"] = chexmask_lookup[dicom_id]
            s_new["chexmask_image_id"] = dicom_id
            matched.append(s_new)
        else:
            missing.append((dicom_id, s))

    return matched, missing

def attach_chexmask_to_temporal_samples(samples, chexmask_lookup):
    matched = []
    missing = []

    for s in samples:
        meta = s["metadata"]
        dicom_id = meta.get("current_dicom_id")

        if dicom_id is None:
            missing.append(("NO_CURRENT_DICOM_ID", s))
            continue

        dicom_id = str(dicom_id).strip()

        if dicom_id in chexmask_lookup:
            s_new = dict(s)
            s_new["chexmask_masks"] = chexmask_lookup[dicom_id]
            s_new["chexmask_image_id"] = dicom_id
            matched.append(s_new)
        else:
            missing.append((dicom_id, s))

    return matched, missing

# =========================================================
# Proxy pathology masks
# =========================================================

def build_lower_lung_mask(
    left_lung_mask: torch.Tensor,
    right_lung_mask: torch.Tensor,
    lower_fraction: float = 0.35,
) -> torch.Tensor:
    """
    Build a lower-lung proxy mask for pleural effusion.
    """
    left = (left_lung_mask > 0).float()
    right = (right_lung_mask > 0).float()
    lung = ((left + right) > 0).float()

    h, _ = lung.shape
    lower = torch.zeros_like(lung)
    lower[int((1.0 - lower_fraction) * h):, :] = 1.0

    return lung * lower


def build_upper_lung_mask(
    left_lung_mask: torch.Tensor,
    right_lung_mask: torch.Tensor,
    upper_fraction: float = 0.30,
) -> torch.Tensor:
    """
    Build an upper-lung proxy mask for pneumothorax.
    """
    left = (left_lung_mask > 0).float()
    right = (right_lung_mask > 0).float()
    lung = ((left + right) > 0).float()

    h, _ = lung.shape
    upper = torch.zeros_like(lung)
    upper[:int(upper_fraction * h), :] = 1.0

    return lung * upper


def build_proxy_mask(
    pathology_name: str,
    masks: dict[str, Any],
) -> torch.Tensor:
    """
    Build a pathology-specific target mask.

    Logic:
        - If the pathology exists directly as a mask key, use it.
        - Otherwise use anatomy-aware proxy rules.

    Supported proxy pathologies:
        - Cardiomegaly -> heart
        - Pleural Effusion / Effusion -> lower lungs
        - Pneumothorax -> upper lungs
    """
    p = pathology_name.lower().strip()

    # direct path if the exact pathology mask exists
    for key in masks.keys():
        if key.lower().strip() == p:
            return (masks[key] > 0).float()

    if p == "cardiomegaly":
        if "heart" not in masks:
            raise KeyError("Need 'heart' mask for cardiomegaly.")
        return (masks["heart"] > 0).float()

    if p in {"pleural effusion", "effusion"}:
        return build_lower_lung_mask(
            masks["left_lung"],
            masks["right_lung"],
        )

    if p == "pneumothorax":
        return build_upper_lung_mask(
            masks["left_lung"],
            masks["right_lung"],
        )

    raise ValueError(f"No proxy mask rule defined for pathology: {pathology_name}")


# =========================================================
# Pointing Game
# =========================================================

def pointing_game_mask(
    heatmap: torch.Tensor,
    gt_mask: torch.Tensor,
) -> dict[str, Any]:
    """
    Standard Pointing Game with a binary mask target.

    A hit occurs if the maximum-activation point of the heatmap
    falls inside the target mask.
    """
    heatmap, gt_mask = ensure_same_size(heatmap, gt_mask)

    row, col = get_peak_point(heatmap)
    hit = bool(gt_mask[row, col].item() > 0)

    return {
        "hit": hit,
        "peak_row": row,
        "peak_col": col,
    }


def evaluate_one_sample_with_mask(
    heatmap: torch.Tensor,
    pathology_name: str,
    chexmask_masks: dict[str, Any],
    image_id: str | None = None,
    method_name: str | None = None,
) -> dict[str, Any]:
    """
    Evaluate one explanation heatmap against a pathology-specific mask.
    """
    gt_mask = build_proxy_mask(pathology_name, chexmask_masks)

    out = pointing_game_mask(heatmap, gt_mask)
    out["pathology_name"] = pathology_name
    if image_id is not None:
        out["image_id"] = image_id
    if method_name is not None:
        out["method_name"] = method_name
    return out


def compute_pointing_score(
    results: Sequence[Optional[dict[str, Any]]],
) -> float:
    """
    Compute dataset-level Pointing Game accuracy.
    """
    valid = [r for r in results if r is not None]
    if not valid:
        return 0.0
    return sum(int(r["hit"]) for r in valid) / len(valid)

def patch_records_with_pathology_masks(records):
    """
    Replace anatomical CheXmask entries with pathology-level proxy masks
    so evaluate_dataset_with_masks can find them by pathology name.
    """
    patched = []
    skipped = 0

    for r in records:
        p = r["pathology_name"]
        chexmask = r["sample"].get("chexmask_masks", {})
        mask = build_pathology_mask(chexmask, p)

        if mask is None:
            skipped += 1
            continue

        # Inject the mask under the pathology name
        r_new = dict(r)
        r_new["sample"] = dict(r["sample"])
        r_new["sample"]["chexmask_masks"] = dict(chexmask)
        r_new["sample"]["chexmask_masks"][p] = mask
        patched.append(r_new)

    print(f"Patched {len(patched)} records, skipped {skipped} (no anatomy mask)")
    return patched


def evaluate_dataset_with_masks(
    explanation_records: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], float]:
    """
    Evaluate a list of explanation records.

    Each record must contain:
        {
            "heatmap": Tensor[H, W],
            "pathology_name": str,
            "chexmask_masks": dict,
        }

    Optional:
        "image_id"
        "method_name"
    """
    results: list[dict[str, Any]] = []

    for rec in explanation_records:
        out = evaluate_one_sample_with_mask(
            heatmap=rec["heatmap"],
            pathology_name=rec["pathology_name"],
            chexmask_masks=rec["chexmask_masks"],
            image_id=rec.get("image_id"),
            method_name=rec.get("method_name"),
        )
        results.append(out)

    score = compute_pointing_score(results)
    return results, score


# =========================================================
# Summaries
# =========================================================

def summarize_results_by_method(results):
    """
    results: list of dicts with at least:
        - "method_name"
        - "hit"
    """
    valid = [r for r in results if r is not None and "method_name" in r]

    if not valid:
        return pd.DataFrame(columns=["method", "n", "hits", "pointing_score"])

    methods = sorted(set(r["method_name"] for r in valid))
    rows = []

    for method in methods:
        subset = [r for r in valid if r["method_name"] == method]
        hits = sum(int(r["hit"]) for r in subset)

        rows.append({
            "method": method,
            "n": len(subset),
            "hits": hits,
            "pointing_score": hits / len(subset),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("pointing_score", ascending=False).reset_index(drop=True)
    return df


def summarize_results_by_pathology(results):
    """
    results: list of dicts with at least:
        - "pathology_name"
        - "hit"
    """
    valid = [r for r in results if r is not None and "pathology_name" in r]

    if not valid:
        return pd.DataFrame(columns=["pathology_name", "n", "hits", "pointing_score"])

    pathologies = sorted(set(r["pathology_name"] for r in valid))
    rows = []

    for pathology in pathologies:
        subset = [r for r in valid if r["pathology_name"] == pathology]
        hits = sum(int(r["hit"]) for r in subset)

        rows.append({
            "pathology_name": pathology,
            "n": len(subset),
            "hits": hits,
            "pointing_score": hits / len(subset),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("pointing_score", ascending=False).reset_index(drop=True)
    return df



    # =========================================================
# Selected-sample Grad-CAM / Grad-CAM++ for fair comparison
# with CXR-CLIP (keeps existing spatial + temporal logic)
# =========================================================

from typing import Mapping
from PIL import Image
import matplotlib.pyplot as plt

from pipelines.biovil_t_pipeline import (
    get_device,
    load_biovil_t,
    load_biovil_preprocessor,
)
from interpretability.biovil_t.attribution.grad_cam import (
    BioViLTGradCAM,
    BioViLTGradCAMPlusPlus,
    find_target_layer,
)


SUPPORTED_SELECTED_CAM_METHODS = ("gradcam", "gradcam++")


def index_single_samples_by_image_path(single_samples):
    return {
        s["metadata"]["image_path"]: s
        for s in single_samples
        if "metadata" in s and "image_path" in s["metadata"]
    }


def match_cases_by_label_to_biovil_t_samples(
    cases_by_label: Dict[str, pd.DataFrame],
    single_samples,
) -> Dict[str, List[dict]]:
    """
    Match selected CXR-CLIP case rows to BioViL-T single samples by image_path.
    """
    index = index_single_samples_by_image_path(single_samples)
    matched: Dict[str, List[dict]] = {}

    for label, rows_df in cases_by_label.items():
        matched[label] = []
        if rows_df is None or len(rows_df) == 0:
            continue

        for _, row in rows_df.iterrows():
            img_path = row["image_path"]
            if img_path not in index:
                continue

            matched[label].append({
                "row": row,
                "sample": index[img_path],
            })

    return matched


def _load_display_image(image_path):
    img = Image.open(image_path).convert("L")
    img = np.asarray(img, dtype=np.float32) / 255.0
    return img


def _prepare_biovil_t_single_inputs_for_cam(
    image_path: str,
    report_text: str,
    image_transform,
    tokenizer,
    device: str,
):
    """
    Uses the BioViL-T repo preprocessors:
    - image_transform is a torchvision-style Compose
    - tokenizer is HF-style
    """
    image = Image.open(image_path).convert("L")

    pixel_values = image_transform(image)
    if pixel_values.ndim == 3:
        pixel_values = pixel_values.unsqueeze(0)
    pixel_values = pixel_values.to(device)

    text_inputs = tokenizer(
        report_text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=256,
    )

    input_ids = text_inputs["input_ids"].to(device)
    attention_mask = text_inputs["attention_mask"].to(device)

    return pixel_values, input_ids, attention_mask


def generate_biovil_t_gradcam_heatmap_for_selected_sample(
    matched_sample: Mapping[str, Any],
    model,
    image_transform,
    tokenizer,
    target_layer,
    device: str,
    method: str = "gradcam",
) -> torch.Tensor:
    """
    Generate Grad-CAM / Grad-CAM++ for one exact selected BioViL-T sample.
    """
    if method not in SUPPORTED_SELECTED_CAM_METHODS:
        raise ValueError(f"Unsupported method: {method}")

    row = matched_sample["row"]
    sample = matched_sample["sample"]

    image_path = row["image_path"]
    report_text = sample["metadata"]["report_text"]

    pixel_values, input_ids, attention_mask = _prepare_biovil_t_single_inputs_for_cam(
        image_path=image_path,
        report_text=report_text,
        image_transform=image_transform,
        tokenizer=tokenizer,
        device=device,
    )

    if method == "gradcam":
        cam_extractor = BioViLTGradCAM(model=model, target_layer=target_layer)
    else:
        cam_extractor = BioViLTGradCAMPlusPlus(model=model, target_layer=target_layer)

    cam, _ = cam_extractor(pixel_values, input_ids, attention_mask)
    return cam[0].float().cpu()


def build_biovil_t_selected_cam_pointing_records(
    cases_by_label: Dict[str, pd.DataFrame],
    single_samples,
    chexmask_lookup: Dict[str, Dict[str, Any]],
    methods: Sequence[str] = ("gradcam",),
    model_name: str = "microsoft/BiomedVLP-BioViL-T",
    device: str | None = None,
    target_layer_name: str | None = None,
):
    """
    Build pointing-game records for selected samples only, using Grad-CAM / Grad-CAM++.
    This is intended for fair same-sample comparison with CXR-CLIP.
    """
    for method in methods:
        if method not in SUPPORTED_SELECTED_CAM_METHODS:
            raise ValueError(
                f"Unsupported method: {method}. Must be one of {SUPPORTED_SELECTED_CAM_METHODS}."
            )

    matched = match_cases_by_label_to_biovil_t_samples(cases_by_label, single_samples)

    if device is None:
        device = get_device()

    model = load_biovil_t(model_name=model_name, device=device)
    model.eval()

    image_transform = load_biovil_preprocessor(model_name=model_name, mode="image")
    tokenizer = load_biovil_preprocessor(model_name=model_name, mode="text")

    target_layer = find_target_layer(
        model.image_model,
        layer_name=target_layer_name,
    )

    records: List[Dict[str, Any]] = []

    for label, matched_items in matched.items():
        for item in matched_items:
            row = item["row"]

            dicom_id = str(row["dicom_id"]).strip()
            if dicom_id not in chexmask_lookup:
                continue

            chexmask_masks = chexmask_lookup[dicom_id]

            for method in methods:
                heatmap = generate_biovil_t_gradcam_heatmap_for_selected_sample(
                    matched_sample=item,
                    model=model,
                    image_transform=image_transform,
                    tokenizer=tokenizer,
                    target_layer=target_layer,
                    device=device,
                    method=method,
                )

                records.append({
                    "heatmap": heatmap,
                    "pathology_name": label,
                    "chexmask_masks": chexmask_masks,
                    "image_id": dicom_id,
                    "method_name": method,
                    "subject_id": row.get("subject_id"),
                    "study_id": row.get("study_id"),
                    "dicom_id": row.get("dicom_id"),
                    "image_path": row.get("image_path"),
                    "true": row.get(f"{label}_true"),
                    "pred": row.get(f"{label}_pred_calibrated"),
                    "margin": row.get(f"{label}_margin"),
                })

    return records


def evaluate_biovil_t_selected_cam_pointing_game(
    cases_by_label: Dict[str, pd.DataFrame],
    single_samples,
    chexmask_lookup: Dict[str, Dict[str, Any]],
    methods: Sequence[str] = ("gradcam",),
    model_name: str = "microsoft/BiomedVLP-BioViL-T",
    device: str | None = None,
    target_layer_name: str | None = None,
):
    records = build_biovil_t_selected_cam_pointing_records(
        cases_by_label=cases_by_label,
        single_samples=single_samples,
        chexmask_lookup=chexmask_lookup,
        methods=methods,
        model_name=model_name,
        device=device,
        target_layer_name=target_layer_name,
    )

    results, overall_score = evaluate_dataset_with_masks(records)

    results_df = pd.DataFrame(results)
    summary_method = summarize_results_by_method(results)
    summary_pathology = summarize_results_by_pathology(results)

    return {
        "records": records,
        "results": results,
        "results_df": results_df,
        "overall_score": overall_score,
        "summary_method": summary_method,
        "summary_pathology": summary_pathology,
    }


def plot_biovil_t_selected_cam_pointing_game_figure(
    cases_by_label: Dict[str, pd.DataFrame],
    single_samples,
    chexmask_lookup: Dict[str, Dict[str, Any]],
    methods: Sequence[str] = ("gradcam",),
    model_name: str = "microsoft/BiomedVLP-BioViL-T",
    device: str | None = None,
    target_layer_name: str | None = None,
    alpha: float = 0.35,
    cmap: str = "magma",
    save_path: str | Path | None = None,
    dpi: int = 300,
):
    for method in methods:
        if method not in SUPPORTED_SELECTED_CAM_METHODS:
            raise ValueError(
                f"Unsupported method: {method}. Must be one of {SUPPORTED_SELECTED_CAM_METHODS}."
            )

    matched = match_cases_by_label_to_biovil_t_samples(cases_by_label, single_samples)

    plot_rows: List[tuple[str, dict]] = []
    for label, items in matched.items():
        for item in items:
            plot_rows.append((label, item))

    n = len(plot_rows)
    if n == 0:
        print("No cases to plot.")
        return

    if device is None:
        device = get_device()

    model = load_biovil_t(model_name=model_name, device=device)
    model.eval()

    image_transform = load_biovil_preprocessor(model_name=model_name, mode="image")
    tokenizer = load_biovil_preprocessor(model_name=model_name, mode="text")
    target_layer = find_target_layer(
        model.image_model,
        layer_name=target_layer_name,
    )

    ncols = 2 + len(methods)  # original + methods + proxy mask
    fig, axes = plt.subplots(
        nrows=n,
        ncols=ncols,
        figsize=(4 * ncols, 4 * n),
        constrained_layout=True,
    )

    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    col_titles = ["Original"] + [m for m in methods] + ["Proxy mask"]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=12)

    for i, (label, item) in enumerate(plot_rows):
        row = item["row"]
        dicom_id = str(row["dicom_id"]).strip()

        if dicom_id not in chexmask_lookup:
            for j in range(ncols):
                axes[i, j].axis("off")
            continue

        chexmask_masks = chexmask_lookup[dicom_id]

        disp_img = _load_display_image(row["image_path"])
        img_h, img_w = disp_img.shape

        axes[i, 0].imshow(disp_img, cmap="gray")
        axes[i, 0].axis("off")

        heatmaps = {}
        peaks = {}

        for j, method in enumerate(methods, start=1):
            heatmap = generate_biovil_t_gradcam_heatmap_for_selected_sample(
                matched_sample=item,
                model=model,
                image_transform=image_transform,
                tokenizer=tokenizer,
                target_layer=target_layer,
                device=device,
                method=method,
            )

            gt_mask = build_proxy_mask(label, chexmask_masks)
            heatmap_eval, gt_mask = ensure_same_size(heatmap, gt_mask)
            peak_row, peak_col = get_peak_point(heatmap_eval)

            heatmaps[method] = heatmap_eval
            peaks[method] = (peak_row, peak_col)

            heatmap_disp = F.interpolate(
                heatmap_eval.unsqueeze(0).unsqueeze(0),
                size=(img_h, img_w),
                mode="bilinear",
                align_corners=False,
            )[0, 0].numpy()

            eval_h, eval_w = heatmap_eval.shape
            disp_peak_row = peak_row * (img_h / eval_h)
            disp_peak_col = peak_col * (img_w / eval_w)

            axes[i, j].imshow(disp_img, cmap="gray")
            axes[i, j].imshow(heatmap_disp, cmap=cmap, alpha=alpha)
            axes[i, j].scatter(disp_peak_col, disp_peak_row, c="cyan", s=50, marker="x")
            axes[i, j].axis("off")

        ref_method = methods[0]
        peak_row, peak_col = peaks[ref_method]
        gt_mask = build_proxy_mask(label, chexmask_masks)
        _, gt_mask = ensure_same_size(heatmaps[ref_method], gt_mask)

        hit = bool(gt_mask[peak_row, peak_col].item() > 0)

        axes[i, -1].imshow(gt_mask.numpy(), cmap="gray")
        axes[i, -1].scatter(peak_col, peak_row, c="red", s=50, marker="x")
        axes[i, -1].axis("off")

        row_meta = (
            f"{label}\n"
            f"subj={row['subject_id']} | dicom={row['dicom_id']}\n"
            f"true={int(row.get(f'{label}_true', 0))} | "
            f"pred={int(row.get(f'{label}_pred_calibrated', 0))} | "
            f"hit={int(hit)}"
        )
        axes[i, 0].set_ylabel(row_meta, fontsize=10)

    fig.suptitle("BioViL-T Selected-Sample Grad-CAM Pointing Game", fontsize=14)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    plt.show()
    plt.close(fig)









def get_sample_from_row(row, test_df, pipeline):
    sample_df = test_df[test_df["image_path"] == row["image_path"]].reset_index(drop=True)
    if len(sample_df) == 0:
        raise ValueError(f"Could not find image_path in test_df: {row['image_path']}")

    sample_dataset = CXRClipInferenceDataset(
        sample_df,
        transform=pipeline["preprocess"],
    )
    return sample_dataset[0]


def generate_cxr_clip_heatmap(
    sample: Mapping[str, Any],
    pipeline: Dict[str, Any],
    label: str,
    method: str = "similarity",
    use_margin: bool = True,
) -> torch.Tensor:
    """
    Generic heatmap generator for CXR-CLIP methods.

    method:
        - 'similarity'
        - 'gradcam'
        - 'gradcam++'
    """
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unsupported method: {method}. Must be one of {SUPPORTED_METHODS}.")

    if method == "similarity":
        heatmap = generate_similarity_relevance_map(
            pipeline=pipeline,
            image_tensor=sample["image"],
            label=label,
            use_margin=use_margin,
        )
    elif method == "gradcam":
        heatmap = generate_gradcam(
            pipeline=pipeline,
            image_tensor=sample["image"],
            label=label,
            use_margin=use_margin,
        )
    else:  # gradcam++
        heatmap = generate_gradcam_plus_plus(
            pipeline=pipeline,
            image_tensor=sample["image"],
            label=label,
            use_margin=use_margin,
        )

    return torch.tensor(heatmap, dtype=torch.float32)


def build_cxr_clip_pointing_records(
    cases_by_label: Dict[str, pd.DataFrame],
    test_df: pd.DataFrame,
    pipeline: Dict[str, Any],
    chexmask_lookup: Dict[str, Dict[str, Any]],
    methods: Sequence[str] = ("similarity", "gradcam"),
    use_margin: bool = True,
) -> List[Dict[str, Any]]:
    """
    Build explanation records in the shared pointing-game format for multiple labels and methods.
    """
    for method in methods:
        if method not in SUPPORTED_METHODS:
            raise ValueError(f"Unsupported method: {method}. Must be one of {SUPPORTED_METHODS}.")

    records: List[Dict[str, Any]] = []

    for label, rows_df in cases_by_label.items():
        if rows_df is None or len(rows_df) == 0:
            continue

        for _, row in rows_df.iterrows():
            dicom_id = str(row["dicom_id"]).strip()
            if dicom_id not in chexmask_lookup:
                continue

            sample = get_sample_from_row(row, test_df, pipeline)
            chexmask_masks = chexmask_lookup[dicom_id]

            for method in methods:
                heatmap = generate_cxr_clip_heatmap(
                    sample=sample,
                    pipeline=pipeline,
                    label=label,
                    method=method,
                    use_margin=use_margin,
                )

                records.append(
                    {
                        "heatmap": heatmap,
                        "pathology_name": label,
                        "chexmask_masks": chexmask_masks,
                        "image_id": dicom_id,
                        "method_name": method,
                        "subject_id": row.get("subject_id"),
                        "study_id": row.get("study_id"),
                        "dicom_id": row.get("dicom_id"),
                        "image_path": row.get("image_path"),
                        "true": row.get(f"{label}_true"),
                        "pred": row.get(f"{label}_pred_calibrated"),
                        "margin": row.get(f"{label}_margin"),
                    }
                )

    return records


def evaluate_cxr_clip_pointing_game(
    cases_by_label: Dict[str, pd.DataFrame],
    test_df: pd.DataFrame,
    pipeline: Dict[str, Any],
    chexmask_lookup: Dict[str, Dict[str, Any]],
    methods: Sequence[str] = ("similarity", "gradcam"),
    use_margin: bool = True,
) -> Dict[str, Any]:
    """
    Run pointing-game evaluation for the provided labels/cases and methods.
    """
    records = build_cxr_clip_pointing_records(
        cases_by_label=cases_by_label,
        test_df=test_df,
        pipeline=pipeline,
        chexmask_lookup=chexmask_lookup,
        methods=methods,
        use_margin=use_margin,
    )

    results, overall_score = evaluate_dataset_with_masks(records)

    results_df = pd.DataFrame(results)
    summary_method = summarize_results_by_method(results)
    summary_pathology = summarize_results_by_pathology(results)

    return {
        "records": records,
        "results": results,
        "results_df": results_df,
        "overall_score": overall_score,
        "summary_method": summary_method,
        "summary_pathology": summary_pathology,
    }


def _load_display_image(image_path):
    """
    Load original grayscale X-ray for display.
    Returns float array [H, W] in [0, 1].
    """
    img = Image.open(image_path).convert("L")
    img = np.asarray(img, dtype=np.float32) / 255.0
    return img


def _resize_heatmap_to_image(heatmap_t: torch.Tensor, image_hw):
    """
    Resize heatmap tensor [H, W] to match display image size.
    image_hw = (H, W)
    """
    if tuple(heatmap_t.shape) == tuple(image_hw):
        return heatmap_t

    return F.interpolate(
        heatmap_t.unsqueeze(0).unsqueeze(0),
        size=image_hw,
        mode="bilinear",
        align_corners=False,
    )[0, 0]


def plot_cxr_clip_pointing_game_figure(
    cases_by_label,
    test_df,
    pipeline,
    chexmask_lookup,
    methods=("similarity", "gradcam"),
    use_margin=True,
    alpha=0.35,
    cmap="magma",
    save_path=None,
    dpi=300,
):
    """
    Plot all selected cases in one figure with correctly aligned original images.

    Columns:
        0 = original image
        1..N = one heatmap overlay per method
        last = proxy mask with peak from first method
    """
    for method in methods:
        if method not in SUPPORTED_METHODS:
            raise ValueError(f"Unsupported method: {method}. Must be one of {SUPPORTED_METHODS}.")

    plot_rows = []
    for label, rows_df in cases_by_label.items():
        if rows_df is None or len(rows_df) == 0:
            continue
        for _, row in rows_df.iterrows():
            plot_rows.append((label, row))

    n = len(plot_rows)
    if n == 0:
        print("No cases to plot.")
        return

    ncols = 2 + len(methods)  # original + methods + proxy mask
    fig, axes = plt.subplots(
        nrows=n,
        ncols=ncols,
        figsize=(4 * ncols, 4 * n),
        constrained_layout=True,
    )

    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    col_titles = ["Original"] + [m for m in methods] + ["Proxy mask"]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=12)

    for i, (label, row) in enumerate(plot_rows):
        dicom_id = str(row["dicom_id"]).strip()
        if dicom_id not in chexmask_lookup:
            for j in range(ncols):
                axes[i, j].axis("off")
            continue

        sample = get_sample_from_row(row, test_df, pipeline)
        chexmask_masks = chexmask_lookup[dicom_id]

        # Load original image for display
        disp_img = _load_display_image(row["image_path"])
        img_h, img_w = disp_img.shape

        axes[i, 0].imshow(disp_img, cmap="gray")
        axes[i, 0].axis("off")

        heatmaps = {}
        peaks = {}

        for j, method in enumerate(methods, start=1):
            heatmap = generate_cxr_clip_heatmap(
                sample=sample,
                pipeline=pipeline,
                label=label,
                method=method,
                use_margin=use_margin,
            )

            # Build proxy mask in its native size for evaluation
            gt_mask = build_proxy_mask(label, chexmask_masks)
            heatmap_eval, gt_mask = ensure_same_size(heatmap, gt_mask)
            peak_row, peak_col = get_peak_point(heatmap_eval)

            heatmaps[method] = heatmap_eval
            peaks[method] = (peak_row, peak_col)

            # Resize heatmap to display image size
            heatmap_disp = _resize_heatmap_to_image(heatmap_eval, (img_h, img_w)).numpy()

            # Map evaluation peak into display coordinates
            eval_h, eval_w = heatmap_eval.shape
            disp_peak_row = peak_row * (img_h / eval_h)
            disp_peak_col = peak_col * (img_w / eval_w)

            axes[i, j].imshow(disp_img, cmap="gray")
            axes[i, j].imshow(heatmap_disp, cmap=cmap, alpha=alpha)
            axes[i, j].scatter(disp_peak_col, disp_peak_row, c="cyan", s=50, marker="x")
            axes[i, j].axis("off")

        # Proxy mask panel uses first method peak
        ref_method = methods[0]
        peak_row, peak_col = peaks[ref_method]
        gt_mask = build_proxy_mask(label, chexmask_masks)
        _, gt_mask = ensure_same_size(heatmaps[ref_method], gt_mask)

        hit = bool(gt_mask[peak_row, peak_col].item() > 0)

        axes[i, -1].imshow(gt_mask.numpy(), cmap="gray")
        axes[i, -1].scatter(peak_col, peak_row, c="red", s=50, marker="x")
        axes[i, -1].axis("off")

        row_meta = (
            f"{label}\n"
            f"subj={row['subject_id']} | dicom={row['dicom_id']}\n"
            f"true={int(row.get(f'{label}_true', 0))} | "
            f"pred={int(row.get(f'{label}_pred_calibrated', 0))} | "
            f"hit={int(hit)}"
        )
        axes[i, 0].set_ylabel(row_meta, fontsize=10)

    fig.suptitle("CXR-CLIP Pointing Game: Selected Cases", fontsize=14)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    plt.show()
    plt.close(fig)