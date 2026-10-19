from __future__ import annotations

import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch


def _add_repo_to_path(repo_root: Union[str, Path]) -> str:
    repo_root = str(Path(repo_root).resolve())
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    return repo_root


def _load_checkpoint(checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
    checkpoint_path = str(Path(checkpoint_path).resolve())

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if not isinstance(ckpt, dict):
        raise TypeError("Checkpoint must be a dictionary-like object.")

    if "config" not in ckpt:
        raise KeyError("Checkpoint missing required key: 'config'")
    if "model" not in ckpt:
        raise KeyError("Checkpoint missing required key: 'model'")

    return ckpt


def _get_writable_hf_cache_dir() -> str:
    cache_dir = (
        os.environ.get("HF_HOME")
        or os.environ.get("HF_HUB_CACHE")
        or os.environ.get("TRANSFORMERS_CACHE")
    )

    if cache_dir is None:
        cache_dir = str(Path.home() / ".cache" / "huggingface")

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    return cache_dir


def _patch_cache_dirs(ckpt_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Override any bad / unwritable Hugging Face cache dirs stored in the checkpoint.
    """
    cfg = deepcopy(ckpt_config)
    hf_cache_dir = _get_writable_hf_cache_dir()

    # tokenizer cache dir
    if "tokenizer" in cfg and cfg["tokenizer"] is not None:
        cfg["tokenizer"]["cache_dir"] = hf_cache_dir

    # text encoder cache dir
    if "model" in cfg and "text_encoder" in cfg["model"]:
        cfg["model"]["text_encoder"]["cache_dir"] = hf_cache_dir

    # also set env vars so downstream HF calls use the same writable location
    os.environ["HF_HOME"] = hf_cache_dir
    os.environ["TRANSFORMERS_CACHE"] = hf_cache_dir
    os.environ["HF_HUB_CACHE"] = hf_cache_dir

    return cfg


def load_cxr_clip(
    checkpoint_path: Union[str, Path],
    repo_root: Union[str, Path] = "models/cxr_clip/cxr-clip",
    device: Optional[str] = None,
) -> Tuple[torch.nn.Module, Any, Dict[str, Any]]:
    _add_repo_to_path(repo_root)

    from cxrclip.model import build_model
    from cxrclip.data.data_utils import load_tokenizer

    ckpt = _load_checkpoint(checkpoint_path)
    ckpt_config = _patch_cache_dirs(ckpt["config"])

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = None
    if "tokenizer" in ckpt_config and ckpt_config["tokenizer"] is not None:
        tokenizer = load_tokenizer(**ckpt_config["tokenizer"])

    model = build_model(
        model_config=ckpt_config["model"],
        loss_config=ckpt_config["loss"],
        tokenizer=tokenizer,
    )

    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)

    model = model.to(device)
    model.eval()

    if len(missing) > 0:
        print(f"[load_cxr_clip] Missing keys: {len(missing)}")
    if len(unexpected) > 0:
        print(f"[load_cxr_clip] Unexpected keys: {len(unexpected)}")

    return model, tokenizer, ckpt_config


@torch.no_grad()
def encode_image(model: torch.nn.Module, image_tensor: torch.Tensor) -> torch.Tensor:
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)

    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)

    img_emb = model.encode_image(image_tensor)
    img_emb = model.image_projection(img_emb) if getattr(model, "projection", False) else img_emb
    img_emb = img_emb / torch.norm(img_emb, dim=1, keepdim=True)
    return img_emb


@torch.no_grad()
def encode_text(
    model: torch.nn.Module,
    tokenizer,
    texts: Union[str, Sequence[str]],
    max_length: int = 77,
) -> torch.Tensor:
    if tokenizer is None:
        raise ValueError("Tokenizer is None. Checkpoint config must include tokenizer settings.")

    if isinstance(texts, str):
        texts = [texts]

    tokens = tokenizer(
        list(texts),
        padding="longest",
        truncation=True,
        return_tensors="pt",
        max_length=max_length,
    )

    device = next(model.parameters()).device
    tokens = {k: v.to(device) for k, v in tokens.items()}

    txt_emb = model.encode_text(tokens)
    txt_emb = model.text_projection(txt_emb) if getattr(model, "projection", False) else txt_emb
    txt_emb = txt_emb / torch.norm(txt_emb, dim=1, keepdim=True)
    return txt_emb


@torch.no_grad()
def score_prompts(
    model: torch.nn.Module,
    tokenizer,
    image_tensor: torch.Tensor,
    prompts: Sequence[str],
    max_length: Optional[int] = None,
) -> torch.Tensor:
    if max_length is None:
        max_length = 77

    img_emb = encode_image(model, image_tensor)
    txt_emb = encode_text(model, tokenizer, prompts, max_length=max_length)

    scores = img_emb @ txt_emb.T
    return scores.squeeze(0).detach().cpu()


@torch.no_grad()
def score_positive_negative_prompt(
    model: torch.nn.Module,
    tokenizer,
    image_tensor: torch.Tensor,
    positive_prompt: str,
    negative_prompt: str,
    max_length: Optional[int] = None,
) -> Dict[str, float]:
    prompts = [positive_prompt, negative_prompt]
    scores = score_prompts(
        model=model,
        tokenizer=tokenizer,
        image_tensor=image_tensor,
        prompts=prompts,
        max_length=max_length,
    )

    pos_score = float(scores[0].item())
    neg_score = float(scores[1].item())

    return {
        "pos_score": pos_score,
        "neg_score": neg_score,
        "margin": pos_score - neg_score,
    }