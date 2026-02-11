# load.py
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import torch
from transformers import AutoTokenizer

# Import from cxr-clip repo (make sure it's on PYTHONPATH or installed)
from cxrclip.data import DataModule
from cxrclip.model import build_model


def build_tokenizer(ckpt_config: Dict[str, Any]):
    """
    Prefer DataModule (matches official evaluator path).
    Fall back to AutoTokenizer using checkpoint config (no guessing).
    """
    # 1) Preferred: DataModule
    try:
        dummy_data_config = {"test": {}}  # avoids dataset loading
        datamodule = DataModule(
            data_config=dummy_data_config,
            dataloader_config=ckpt_config.get("dataloader", {}),
            tokenizer_config=ckpt_config.get("tokenizer", None),
            transform_config=ckpt_config.get("transform", None),
        )
        if getattr(datamodule, "tokenizer", None) is None:
            raise RuntimeError("DataModule tokenizer is None.")
        return datamodule.tokenizer
    except Exception:
        pass

    # 2) Fallback: read tokenizer name directly from checkpoint config
    tok_cfg = ckpt_config.get("tokenizer", {}) or {}

    name = tok_cfg.get("pretrained_model_name_or_path")
    if not name:
        raise RuntimeError(
            "Could not initialize tokenizer: checkpoint config missing "
            "'tokenizer.pretrained_model_name_or_path' (or 'tokenizer.name')."
        )

    # Optional: respect cache_dir if present, but do NOT use /data on macOS
    cache_dir = tok_cfg.get("cache_dir", None)
    return AutoTokenizer.from_pretrained(name, cache_dir=cache_dir)


def load_cxr_clip(
    checkpoint_path: str | Path,
    device: Optional[str] = None,
) -> Tuple[torch.nn.Module, Any, Dict[str, Any]]:
    """
    Loads a CXR-CLIP checkpoint for inference.

    Returns: (model, tokenizer, ckpt_config)
    """
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    device_t = torch.device(device)

    # IMPORTANT: weights_only=False for these research checkpoints
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "config" not in ckpt or "model" not in ckpt:
        raise KeyError(
            f"Checkpoint does not look like CXR-CLIP. Expected keys 'config' and 'model'. "
            f"Got keys: {list(ckpt.keys())}"
        )

    ckpt_config = ckpt["config"]

    # Fix common issue: checkpoint config may hardcode HF cache_dir to /data/...
    # (Safe to override to user cache for local machines)
    tok_cfg = ckpt_config.get("tokenizer", {}) or {}
    if isinstance(tok_cfg, dict) and "cache_dir" in tok_cfg and str(tok_cfg["cache_dir"]).startswith("/data"):
        tok_cfg["cache_dir"] = str(Path.home() / ".cache" / "huggingface" / "hub")
        ckpt_config["tokenizer"] = tok_cfg

    text_enc = ckpt_config.get("model", {}).get("text_encoder", {})
    if isinstance(text_enc, dict) and "cache_dir" in text_enc and str(text_enc["cache_dir"]).startswith("/data"):
        text_enc["cache_dir"] = str(Path.home() / ".cache" / "huggingface" / "hub")
        ckpt_config["model"]["text_encoder"] = text_enc

    # Build tokenizer & model
    tokenizer = build_tokenizer(ckpt_config)

    model = build_model(
        model_config=ckpt_config["model"],
        loss_config=ckpt_config.get("loss", {}),
        tokenizer=tokenizer,
    )

    # Load weights
    model.load_state_dict(ckpt["model"], strict=False)

    # Ready for inference
    model.to(device_t)
    model.eval()

    return model, tokenizer, ckpt_config
