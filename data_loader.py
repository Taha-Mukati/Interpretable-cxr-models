# dataloader.py
from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader


def _load_pil(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


class SingleInferenceDataset(Dataset):
    """
    Expected columns in CSV:
      - image_path
      - report_text (or you can swap to report_path and read from file)
      - subject_id, study_id, dicom_id (optional but nice to keep)
      - labels... (optional)
    """
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        if "image_path" not in self.df.columns:
            raise ValueError("Single CSV must contain `image_path`.")
        if "report_text" not in self.df.columns:
            # optional, but you asked to preprocess text - so we expect it
            raise ValueError("Single CSV must contain `report_text` for text preprocessing.")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.df.iloc[idx]
        image = _load_pil(r["image_path"])
        return {
            "image": image,                        # PIL
            "text": str(r["report_text"]),         # raw text
            "image_path": str(r["image_path"]),
            "subject_id": int(r["subject_id"]) if "subject_id" in r else None,
            "study_id": int(r["study_id"]) if "study_id" in r else None,
            "dicom_id": str(r["dicom_id"]) if "dicom_id" in r else None,
        }


class TemporalDataset(Dataset):
    """
    Expected columns in CSV (after you created paths):
      - prior_image_path
      - current_image_path
      - current_report_text (or report_path)
      - subject_id, prior_study_id, current_study_id (optional)
      - gap_days (optional)
      - labels... (optional)
    """
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        required = {"prior_image_path", "current_image_path", "current_report_text"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Temporal CSV missing required columns: {missing}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.df.iloc[idx]
        prior_img = _load_pil(r["prior_image_path"])
        current_img = _load_pil(r["current_image_path"])
        return {
            "prior_image": prior_img,                    # PIL
            "current_image": current_img,                # PIL
            "text": str(r["current_report_text"]),       # raw text
            "prior_image_path": str(r["prior_image_path"]),
            "current_image_path": str(r["current_image_path"]),
            "subject_id": int(r["subject_id"]) if "subject_id" in r else None,
            "gap_days": float(r["gap_days"]) if "gap_days" in r else None,
        }