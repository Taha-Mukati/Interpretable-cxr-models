from __future__ import annotations

import sys
import os
import torch
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from transformers import AutoTokenizer
from torchvision import transforms


# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Path to cxr-clip repo (the folder that contains `cxrclip/`)
CXR_CLIP_REPO = PROJECT_ROOT / "models" / "cxr_clip" / "cxr-clip"

if str(CXR_CLIP_REPO) not in sys.path:
    sys.path.insert(0, str(CXR_CLIP_REPO))


from cxrclip.data import build_transform


def cxr_clip_preprocessor(ckpt_config: dict):

    transform_cfg = ckpt_config.get("transform", None)
    if transform_cfg is None:
        raise RuntimeError("No Transform config found in checkpoint file.")
    
    return build_transform(transform_cfg, split="test")
    
