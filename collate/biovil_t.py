from typing import List, Dict, Any
from pathlib import Path

import pandas as pd
import torch
from health_multimodal.image.data.io import load_image


BIOVIL_T_MAX_LENGTH = 512


def _clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    return str(text).strip().rstrip("!?.")


def collate_biovil_single(
    batch: List[Dict[str, Any]],
    image_processor,
    tokenizer,
):
    images = torch.stack(
        [image_processor(load_image(Path(b["image_path"]))) for b in batch],
        dim=0,
    )

    texts = [_clean_text(b["report_text"]) for b in batch]

    text_inputs = tokenizer.batch_encode_plus(
        batch_text_or_text_pairs=texts,
        add_special_tokens=True,
        padding="longest",
        truncation=True,
        max_length=BIOVIL_T_MAX_LENGTH,
        return_tensors="pt",
    )

    out = {
        "pixel_values": images,
        "input_ids": text_inputs["input_ids"],
        "attention_mask": text_inputs["attention_mask"],
        "meta": {
            "image_path": [b["image_path"] for b in batch],
            "subject_id": [b.get("subject_id") for b in batch],
            "study_id": [b.get("study_id") for b in batch],
            "dicom_id": [b.get("dicom_id") for b in batch],
            "report_path": [b.get("report_path") for b in batch],
            "report_text": texts,
        },
    }

    label_cols = [
        "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
        "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion",
        "Lung Opacity", "No Finding", "Pleural Effusion",
        "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices",
    ]
    present = [c for c in label_cols if c in batch[0]]
    if present:
        out["labels"] = {
            c: torch.tensor([b[c] for b in batch], dtype=torch.float32)
            for c in present
        }

    return out


def collate_biovil_temporal(
    batch: List[Dict[str, Any]],
    image_processor,
    tokenizer,
):
    prior_images = torch.stack(
        [image_processor(load_image(Path(b["prior_image_path"]))) for b in batch],
        dim=0,
    )
    current_images = torch.stack(
        [image_processor(load_image(Path(b["current_image_path"]))) for b in batch],
        dim=0,
    )

    texts = [_clean_text(b["current_report_text"]) for b in batch]

    text_inputs = tokenizer.batch_encode_plus(
        batch_text_or_text_pairs=texts,
        add_special_tokens=True,
        padding="longest",
        truncation=True,
        max_length=BIOVIL_T_MAX_LENGTH,
        return_tensors="pt",
    )

    out = {
        "prior_pixel_values": prior_images,
        "current_pixel_values": current_images,
        "input_ids": text_inputs["input_ids"],
        "attention_mask": text_inputs["attention_mask"],
        "meta": {
            "prior_image_path": [b["prior_image_path"] for b in batch],
            "current_image_path": [b["current_image_path"] for b in batch],
            "subject_id": [b.get("subject_id") for b in batch],
            "prior_study_id": [b.get("prior_study_id") for b in batch],
            "prior_dicom_id": [b.get("prior_dicom_id") for b in batch],
            "current_study_id": [b.get("current_study_id") for b in batch],
            "current_dicom_id": [b.get("current_dicom_id") for b in batch],
            "gap_days": [b.get("gap_days") for b in batch],
            "report_path": [b.get("report_path") for b in batch],
            "current_report_text": texts,
        },
    }

    label_cols = [
        "Pneumothorax", "Pleural Effusion", "Support Devices",
        "Consolidation", "Cardiomegaly", "Atelectasis", "Edema",
        "No Finding", "Enlarged Cardiomediastinum", "Fracture",
        "Lung Lesion", "Pleural Other", "Pneumonia",
    ]
    present = [c for c in label_cols if c in batch[0]]
    if present:
        out["labels"] = {
            c: torch.tensor([b[c] for b in batch], dtype=torch.float32)
            for c in present
        }

    return out