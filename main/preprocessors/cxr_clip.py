from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable, Optional, Union

import torch
from PIL import Image


def _add_repo_to_path(repo_root: Union[str, Path]) -> str:
    """
    Add the official cxr-clip repository root to sys.path.

    Expected repo structure:
        repo_root/
            cxrclip/
            configs/
            train.py
    """
    repo_root = str(Path(repo_root).resolve())
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    return repo_root


def _load_checkpoint_config(checkpoint_path: Union[str, Path]) -> dict:
    """
    Load checkpoint and return the stored config dict.
    """
    checkpoint_path = str(Path(checkpoint_path).resolve())
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if not isinstance(ckpt, dict):
        raise TypeError("Checkpoint must be a dict-like object.")

    if "config" not in ckpt:
        raise KeyError("Checkpoint missing required key: 'config'")

    cfg = ckpt["config"]
    if not isinstance(cfg, dict):
        raise TypeError("Checkpoint 'config' must be a dictionary.")

    return cfg


def _infer_normalization_from_config(cfg: dict) -> str:
    """
    Match official CXR-CLIP behavior:
    - resnet image encoder -> imagenet normalization
    - otherwise -> huggingface normalization
    """
    try:
        image_encoder_name = cfg["model"]["image_encoder"]["name"]
    except KeyError as e:
        raise KeyError(
            "Could not find cfg['model']['image_encoder']['name'] in checkpoint config."
        ) from e

    return "imagenet" if image_encoder_name == "resnet" else "huggingface"


def load_cxr_clip_preprocessor(
    checkpoint_path: Union[str, Path],
    repo_root: Union[str, Path],
    split: str = "test",
) -> Callable[[Image.Image], torch.Tensor]:
    """
    Build an image preprocessor using the official CXR-CLIP repo utilities
    and the transform config stored inside the checkpoint.

    Parameters
    ----------
    checkpoint_path:
        Path to official checkpoint, e.g. model-best.tar
    repo_root:
        Path to local clone of the official cxr-clip repo
    split:
        One of: 'train', 'valid', 'test', 'aug'
        For inference, use 'test'.

    Returns
    -------
    preprocess:
        Callable that takes a PIL image and returns a normalized torch.Tensor
        shaped [3, H, W].
    """
    if split not in {"train", "valid", "test", "aug"}:
        raise ValueError("split must be one of {'train', 'valid', 'test', 'aug'}")

    _add_repo_to_path(repo_root)

    # Official repo helpers
    from cxrclip.data.data_utils import load_transform, transform_image

    cfg = _load_checkpoint_config(checkpoint_path)
    transform_config = cfg.get("transform", None)
    normalize = _infer_normalization_from_config(cfg)

    image_transforms = load_transform(
        split=split,
        transform_config=transform_config,
    )

    def preprocess(image: Image.Image) -> torch.Tensor:
        if not isinstance(image, Image.Image):
            raise TypeError(
                f"preprocess expects PIL.Image.Image, got {type(image).__name__}"
            )

        # Ensure 3-channel RGB input for the model
        image = image.convert("RGB")

        tensor = transform_image(
            image_transforms=image_transforms,
            image=image,
            normalize=normalize,
        )

        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"Expected transform_image(...) to return torch.Tensor, got {type(tensor).__name__}"
            )

        return tensor

    return preprocess


def preprocess_image_path(
    image_path: Union[str, Path],
    preprocess: Callable[[Image.Image], torch.Tensor],
) -> torch.Tensor:
    """
    Open an image from disk and preprocess it.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with Image.open(image_path) as img:
        img = img.convert("RGB")
        return preprocess(img)


def load_preprocess_and_config(
    checkpoint_path: Union[str, Path],
    repo_root: Union[str, Path],
    split: str = "test",
) -> tuple[Callable[[Image.Image], torch.Tensor], dict]:
    """
    Convenience helper that returns both:
    - preprocess callable
    - checkpoint config
    """
    preprocess = load_cxr_clip_preprocessor(
        checkpoint_path=checkpoint_path,
        repo_root=repo_root,
        split=split,
    )
    cfg = _load_checkpoint_config(checkpoint_path)
    return preprocess, cfg