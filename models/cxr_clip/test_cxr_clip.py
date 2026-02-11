# quick_test_cxr_clip.py
# Run: python quick_test_cxr_clip.py
#
# You ONLY need to edit the 4 PATH/TEXT variables below.


from __future__ import annotations

import os
from pathlib import Path

os.environ["HF_HUB_DISABLE_XET"] = "1"


# Force HuggingFace + Transformers cache to a writable location (macOS)
HF_DIR = Path.home() / ".cache" / "huggingface"
os.environ["HF_HOME"] = str(HF_DIR)
os.environ["HF_HUB_CACHE"] = str(HF_DIR / "hub")
os.environ["TRANSFORMERS_CACHE"] = str(HF_DIR / "transformers")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_DIR / "hub")  # some versions use this
os.environ["TORCH_HOME"] = str(Path.home() / ".cache" / "torch")

HF_DIR.mkdir(parents=True, exist_ok=True)
(HF_DIR / "hub").mkdir(parents=True, exist_ok=True)
(HF_DIR / "transformers").mkdir(parents=True, exist_ok=True)

import sys
from pathlib import Path

import torch
from PIL import Image


# ----------------------------
# EDIT THESE 4 VALUES
# ----------------------------
CXR_CLIP_REPO_DIR = Path("models/cxr_clip/cxr-clip")          # repo root (contains cxrclip/, configs/, train.py, etc.)
CHECKPOINT_PATH   = Path("/Users/taha/Desktop/UNI/ECM 3401/interpretable-cxr-models/models/cxr_clip/r50_m.tar")  # extracted from r50_m.tar
IMAGE_PATH        = Path("/Users/taha/Desktop/UNI/ECM 3401/interpretable-cxr-models/test_data/image/p10000980/s57861150/83ccefac-17d59b93-56df282a-68ec74f3-32c3efdf.jpg")
PROMPT            = "Severe cardiomegaly"
# ----------------------------


def _infer_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _get_transform_from_repo_or_fallback(ckpt_config: dict):
    """
    Tries to use the repo's transform builder if available.
    Falls back to a standard CLIP-like image pipeline (Resize -> CenterCrop -> ToTensor -> ImageNet normalize).
    """
    # 1) Try repo transform builder if it exists
    try:
        # Some repos expose a build_transform in cxrclip.data
        from cxrclip.data import build_transform  # type: ignore

        transform_cfg = ckpt_config.get("transform", None)
        if transform_cfg is None:
            raise RuntimeError("No transform config found in checkpoint.")
        return build_transform(transform_cfg, split="test")
    except Exception:
        pass

    # 2) Fallback torchvision transform
    from torchvision import transforms

    # Try to guess size from config
    tcfg = ckpt_config.get("transform", {}) or {}
    size = (
        tcfg.get("image_size")
        or tcfg.get("img_size")
        or tcfg.get("size")
        or tcfg.get("resolution")
        or 224
    )
    if isinstance(size, (list, tuple)) and len(size) > 0:
        size = size[0]
    size = int(size)

    # If ResNet, evaluator forces ImageNet normalization
    # We'll always use ImageNet normalization as a safe default for r50_m.
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)

    return transforms.Compose(
        [
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )


def _build_tokenizer_from_datamodule_or_fallback(ckpt_config: dict):
    """
    Mirrors evaluator.py behavior: uses DataModule to create tokenizer if possible.
    If DataModule requires dataset config and fails, falls back to AutoTokenizer using best-effort config keys.
    """
    # 1) Preferred: DataModule (matches official code path)
    try:
        from cxrclip.data import DataModule  # type: ignore

        dummy_data_config = {"test": {}}  # minimal placeholder; tokenizer init may still work
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

    # 2) Fallback: HuggingFace AutoTokenizer
    from transformers import AutoTokenizer

    tok_cfg = ckpt_config.get("tokenizer", {}) or {}

    # Best-effort common keys
    name = (
        tok_cfg.get("name")
        or tok_cfg.get("model_name")
        or tok_cfg.get("pretrained_model_name_or_path")
        or tok_cfg.get("pretrained")
    )
    if not name:
        raise RuntimeError(
            "Could not initialize tokenizer. DataModule init failed and checkpoint config "
            "does not include a recognizable tokenizer model name."
        )

    return AutoTokenizer.from_pretrained(name)


def main():
    # Make cxr-clip repo importable
    repo_dir = CXR_CLIP_REPO_DIR.resolve()
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))

    device = _infer_device()
    print(f"[CXR-CLIP] Using device: {device}")

    # Load checkpoint (the inner model-best.tar, not the outer r50_m.tar)
    ckpt_path = CHECKPOINT_PATH.resolve()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if "config" not in ckpt or "model" not in ckpt:
        raise KeyError(
            f"Checkpoint does not look like CXR-CLIP. Expected keys 'config' and 'model'. "
            f"Got keys: {list(ckpt.keys())}"
        )

    ckpt_config = ckpt["config"]

    from pathlib import Path

    # Force HF cache away from /data (read-only on macOS)
    hf_cache = str(Path.home() / ".cache" / "huggingface" / "hub")
    (Path.home() / ".cache" / "huggingface" / "hub").mkdir(parents=True, exist_ok=True)

    # Print what was in the checkpoint (useful once)
    print("Before override text_encoder:", ckpt_config.get("model", {}).get("text_encoder", None))

    # Override common places
    if "model" in ckpt_config and "text_encoder" in ckpt_config["model"]:
        ckpt_config["model"]["text_encoder"]["cache_dir"] = hf_cache

    # Some configs nest it differently; try a couple safe fallbacks
    if "model" in ckpt_config and "text" in ckpt_config["model"] and isinstance(ckpt_config["model"]["text"], dict):
        ckpt_config["model"]["text"]["cache_dir"] = hf_cache

    print("After override text_encoder:", ckpt_config.get("model", {}).get("text_encoder", None))



    # Build tokenizer + model (matches evaluator.py)
    tokenizer = _build_tokenizer_from_datamodule_or_fallback(ckpt_config)

    from cxrclip.model import build_model  # type: ignore

    model = build_model(
        model_config=ckpt_config["model"],
        loss_config=ckpt_config.get("loss", {}),
        tokenizer=tokenizer,
    )
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(device)
    model.eval()

    # Build image transform
    transform = _get_transform_from_repo_or_fallback(ckpt_config)

    # Load and preprocess image
    img = Image.open(IMAGE_PATH).convert("RGB")
    image_tensor = transform(img).unsqueeze(0).to(device)  # [1, C, H, W]

    # Tokenize prompt
    max_len = int(ckpt_config.get("base", {}).get("text_max_length", 256))
    text_tokens = tokenizer(
        PROMPT,
        padding="longest",
        truncation=True,
        return_tensors="pt",
        max_length=max_len,
    )
    # Move tokens to device (works for dict-like HF tokenizers)
    if isinstance(text_tokens, dict):
        text_tokens = {k: v.to(device) for k, v in text_tokens.items()}
    else:
        text_tokens = text_tokens.to(device)

    # Encode
    with torch.no_grad():
        # Image embedding
        img_emb = model.encode_image(image_tensor)
        if getattr(model, "projection", False):
            img_emb = model.image_projection(img_emb)
        img_emb = img_emb / img_emb.norm(dim=1, keepdim=True)

        # Text embedding
        txt_emb = model.encode_text(text_tokens)
        if getattr(model, "projection", False):
            txt_emb = model.text_projection(txt_emb)
        txt_emb = txt_emb / txt_emb.norm(dim=1, keepdim=True)

        # Cosine similarity (since both normalized)
        similarity = (img_emb @ txt_emb.T).item()
    
    prompts = [
    "Severe cardiomegaly",
    "Marked cardiomegaly",
    "Enlarged cardiac silhouette",
    "No cardiomegaly"
    "Normal heart size"
]

    for p in prompts:
        text_tokens = tokenizer(
            p,
            padding="longest",
            truncation=True,
            return_tensors="pt",
            max_length=int(ckpt_config.get("base", {}).get("text_max_length", 256)),
        )
        text_tokens = {k: v.to(device) for k, v in text_tokens.items()}

        with torch.no_grad():
            txt_emb = model.encode_text(text_tokens)
            if getattr(model, "projection", False):
                txt_emb = model.text_projection(txt_emb)
            txt_emb = txt_emb / txt_emb.norm(dim=1, keepdim=True)

            sim = (img_emb @ txt_emb.T).item()

        print(f"{p} -> {sim:.6f}")

    print("\n--- Quick CXR-CLIP Test ---")
    print(f"Image:  {IMAGE_PATH}")
    print(f"Prompt: {PROMPT}")
    print(f"Cosine similarity: {similarity:.6f}")
    print("---------------------------\n")


if __name__ == "__main__":
    main()
