from __future__ import annotations

from typing import Any, Dict
import pandas as pd
from torch.utils.data import Dataset
import torch
import numpy as np
from PIL import Image




TARGETS = [
    "Pneumothorax",
    "Pleural Effusion",
    "Support Devices",
    "Consolidation",
    "Cardiomegaly",
    "Atelectasis",
    "Edema",
    "No Finding",
]




def _safe_int(value):
    if pd.isna(value):
        return None
    return int(value)


def _safe_float(value):
    if pd.isna(value):
        return None
    return float(value)


def _safe_str(value):
    if pd.isna(value):
        return None
    return str(value)


def _safe_text(value):
    if pd.isna(value):
        return ""
    return str(value)


class SingleInferenceDataset(Dataset):
    """
    Expected columns:
      - image_path
      - report_text
      - subject_id, study_id, dicom_id (optional)
      - report_path (optional)
      - label columns (optional)
    """

    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)

        required = {"image_path", "report_text"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Single CSV missing required columns: {missing}")

        self.label_cols = [
            "Atelectasis",
            "Cardiomegaly",
            "Consolidation",
            "Edema",
            "Enlarged Cardiomediastinum",
            "Fracture",
            "Lung Lesion",
            "Lung Opacity",
            "No Finding",
            "Pleural Effusion",
            "Pleural Other",
            "Pneumonia",
            "Pneumothorax",
            "Support Devices",
        ]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.df.iloc[idx]

        item = {
            "image_path": str(r["image_path"]),
            "report_text": _safe_text(r["report_text"]),
            "subject_id": _safe_int(r["subject_id"]) if "subject_id" in self.df.columns else None,
            "study_id": _safe_int(r["study_id"]) if "study_id" in self.df.columns else None,
            "dicom_id": _safe_str(r["dicom_id"]) if "dicom_id" in self.df.columns else None,
            "report_path": _safe_str(r["report_path"]) if "report_path" in self.df.columns else None,
        }

        for col in self.label_cols:
            if col in self.df.columns:
                item[col] = float(r[col]) if not pd.isna(r[col]) else 0.0

        return item



class CXRClipInferenceDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True).copy()
        self.transform = transform

        self.df[TARGETS] = self.df[TARGETS].fillna(0).astype(np.float32)

        abnormal = [c for c in TARGETS if c != "No Finding"]
        self.df.loc[self.df[abnormal].sum(axis=1) > 0, "No Finding"] = 0.0

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = Image.open(row["image_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)

        labels = torch.tensor(row[TARGETS].values.astype(np.float32))

        return {
            "image": image,
            "labels": labels,
            "image_path": row["image_path"],
            "study_id": row["study_id"],
            "subject_id": row["subject_id"],
            "dicom_id": row["dicom_id"],
        }




class TemporalDataset(Dataset):
    """
    Expected columns:
      - prior_image_path
      - current_image_path
      - current_report_text
      - subject_id, prior_study_id, prior_dicom_id, current_study_id, current_dicom_id (optional)
      - gap_days (optional)
      - report_path (optional)
      - label columns (optional)
    """

    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)

        required = {"prior_image_path", "current_image_path", "current_report_text"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Temporal CSV missing required columns: {missing}")

        self.label_cols = [
            "Pneumothorax",
            "Pleural Effusion",
            "Support Devices",
            "Consolidation",
            "Cardiomegaly",
            "Atelectasis",
            "Edema",
            "No Finding",
            "Enlarged Cardiomediastinum",
            "Fracture",
            "Lung Lesion",
            "Pleural Other",
            "Pneumonia",
        ]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.df.iloc[idx]

        item = {
            "prior_image_path": str(r["prior_image_path"]),
            "current_image_path": str(r["current_image_path"]),
            "current_report_text": _safe_text(r["current_report_text"]),
            "subject_id": _safe_int(r["subject_id"]) if "subject_id" in self.df.columns else None,
            "prior_study_id": _safe_int(r["prior_study_id"]) if "prior_study_id" in self.df.columns else None,
            "prior_dicom_id": _safe_str(r["prior_dicom_id"]) if "prior_dicom_id" in self.df.columns else None,
            "current_study_id": _safe_int(r["current_study_id"]) if "current_study_id" in self.df.columns else None,
            "current_dicom_id": _safe_str(r["current_dicom_id"]) if "current_dicom_id" in self.df.columns else None,
            "gap_days": _safe_float(r["gap_days"]) if "gap_days" in self.df.columns else None,
            "report_path": _safe_str(r["report_path"]) if "report_path" in self.df.columns else None,
        }

        for col in self.label_cols:
            if col in self.df.columns:
                item[col] = float(r[col]) if not pd.isna(r[col]) else 0.0

        return item