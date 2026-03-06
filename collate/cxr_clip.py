

from __future__ import annotations

from typing import List, Dict, Any
import torch

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CXR_CLIP_REPO = PROJECT_ROOT / "models" / "cxr_clip" / "cxr-clip"

if str(CXR_CLIP_REPO) not in sys.path:
    sys.path.insert(0, str(CXR_CLIP_REPO))

import cxrclip.data.data_utils as du

def collate_cxr_clip_single(
    batch: List[Dict[str, Any]],
    image_transforms,
    normalize_mode: str = "imagenet"
):
    images = torch.stack([
        du.transform_image(image_transforms, b["image"].convert("RGB"), normalize=normalize_mode)
        for b in batch
    ], dim=0)

    texts = [b["text"] for b in batch]

    return {
        "image": images,
        "text": texts,
        "meta": {
            "image_path": [b.get("image_path") for b in batch],
            "subject_id": [b.get("subject_id") for b in batch],
            "study_id": [b.get("study_id") for b in batch],
            "dicom_id": [b.get("dicom_id") for b in batch],
        }
    }