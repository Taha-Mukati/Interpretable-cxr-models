from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CXR_CLIP_REPO = PROJECT_ROOT / "models" / "cxr_clip" / "cxr-clip"

if str(CXR_CLIP_REPO) not in sys.path:
    sys.path.insert(0, str(CXR_CLIP_REPO))

import cxrclip.data.data_utils as du



def _cfg_get(cfg: Any, key: str, default=None):
    try:
        return cfg.get(key, default)
    except Exception:
        return default


def load_cxr_clip_preprocessor(ckpt: dict):
    """
    Returns:
        image_transforms: list of transforms from repo config
        normalize_mode: usually 'imagenet'
    """
    cfg = ckpt["config"]

    transform_cfg = _cfg_get(cfg, "transform", None)
    if transform_cfg is None:
        raise RuntimeError("No transform config found in checkpoint['config'].")

    image_transforms = du.load_transform(split="test", transform_config=transform_cfg)

    normalize_mode = "imagenet"
    data_valid = _cfg_get(cfg, "data_valid", None)
    if data_valid:
        mimic_cfg = _cfg_get(data_valid, "mimic_cxr", None)
        if mimic_cfg:
            normalize_mode = _cfg_get(mimic_cfg, "normalize", "imagenet")

    return image_transforms, normalize_mode