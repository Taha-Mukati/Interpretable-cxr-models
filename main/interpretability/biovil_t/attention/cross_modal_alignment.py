from __future__ import annotations

from typing import List, Sequence

import torch


def _ensure_patch_grid(patch_proj: torch.Tensor) -> torch.Tensor:
    """
    Accepts:
      [H, W, D] or [P, D]
    Returns:
      [H, W, D]
    """
    if patch_proj.ndim == 3:
        return patch_proj

    if patch_proj.ndim == 2:
        p, d = patch_proj.shape
        grid = int(p ** 0.5)
        if grid * grid != p:
            raise ValueError(f"Cannot reshape {p} patches into a square grid")
        return patch_proj.view(grid, grid, d)

    raise ValueError(f"Expected [H,W,D] or [P,D], got {patch_proj.shape}")


def _apply_mask(token_proj: torch.Tensor, tokens: List[str], attention_mask: torch.Tensor | None = None):
    """
    token_proj: [T, D]
    tokens: list[str] of length T or full token list
    attention_mask: [T]
    """
    if attention_mask is None:
        return token_proj, tokens

    mask = attention_mask.bool()
    token_proj = token_proj[mask]
    tokens = [tok for tok, keep in zip(tokens, mask.tolist()) if keep]
    return token_proj, tokens


def compute_alignment_map(
    patch_proj: torch.Tensor,
    text_token_proj: torch.Tensor,
    tokens: List[str],
    attention_mask: torch.Tensor | None = None,
    normalize: bool = True,
) -> tuple[torch.Tensor, List[str]]:
    """
    patch_proj:
      [H, W, D] or [P, D]

    text_token_proj:
      [T, D]

    returns:
      alignment_map: [H, W, T_valid]
      valid_tokens:  list[str]
    """
    patch_grid = _ensure_patch_grid(patch_proj)   # [H, W, D]
    token_proj, valid_tokens = _apply_mask(text_token_proj, tokens, attention_mask)

    if normalize:
        patch_grid = torch.nn.functional.normalize(patch_grid, dim=-1)
        token_proj = torch.nn.functional.normalize(token_proj, dim=-1)

    # [H, W, D] x [T, D] -> [H, W, T]
    alignment_map = torch.einsum("hwd,td->hwt", patch_grid, token_proj)
    return alignment_map, valid_tokens


def token_heatmap(
    patch_proj: torch.Tensor,
    text_token_proj: torch.Tensor,
    tokens: List[str],
    token_index: int,
    attention_mask: torch.Tensor | None = None,
    normalize: bool = False,
) -> tuple[torch.Tensor, str]:
    """
    returns:
      heatmap: [H, W]
      token:   str
    """
    alignment_map, valid_tokens = compute_alignment_map(
        patch_proj=patch_proj,
        text_token_proj=text_token_proj,
        tokens=tokens,
        attention_mask=attention_mask,
        normalize=normalize,
    )

    if token_index < 0 or token_index >= len(valid_tokens):
        raise IndexError(f"token_index {token_index} out of range for {len(valid_tokens)} valid tokens")

    return alignment_map[:, :, token_index], valid_tokens[token_index]


def find_subsequence(sequence: Sequence[str], subsequence: Sequence[str]) -> List[List[int]]:
    matches: List[List[int]] = []
    n, m = len(sequence), len(subsequence)

    if m == 0 or m > n:
        return matches

    for i in range(n - m + 1):
        if list(sequence[i:i + m]) == list(subsequence):
            matches.append(list(range(i, i + m)))
    return matches


def phrase_heatmap(
    patch_proj: torch.Tensor,
    text_token_proj: torch.Tensor,
    tokens: List[str],
    phrase_tokens: Sequence[str],
    attention_mask: torch.Tensor | None = None,
    normalize: bool = False,
    reduce: str = "mean",
) -> tuple[torch.Tensor, List[int]]:
    """
    returns:
      heatmap: [H, W]
      matched_token_indices: list[int]
    """
    alignment_map, valid_tokens = compute_alignment_map(
        patch_proj=patch_proj,
        text_token_proj=text_token_proj,
        tokens=tokens,
        attention_mask=attention_mask,
        normalize=normalize,
    )

    matches = find_subsequence(valid_tokens, phrase_tokens)
    if not matches:
        raise ValueError(f"Phrase tokens {list(phrase_tokens)} not found in token list")

    idxs = matches[0]
    phrase_cube = alignment_map[:, :, idxs]  # [H, W, K]

    if reduce == "mean":
        heatmap = phrase_cube.mean(dim=-1)
    elif reduce == "max":
        heatmap = phrase_cube.max(dim=-1).values
    else:
        raise ValueError("reduce must be 'mean' or 'max'")

    return heatmap, idxs


def topk_tokens_for_patch(
    patch_proj: torch.Tensor,
    text_token_proj: torch.Tensor,
    tokens: List[str],
    patch_row: int,
    patch_col: int,
    attention_mask: torch.Tensor | None = None,
    k: int = 10,
    normalize: bool = False,
):
    """
    returns top-k aligned tokens for one patch location
    """
    alignment_map, valid_tokens = compute_alignment_map(
        patch_proj=patch_proj,
        text_token_proj=text_token_proj,
        tokens=tokens,
        attention_mask=attention_mask,
        normalize=normalize,
    )

    scores = alignment_map[patch_row, patch_col]  # [T]
    vals, idxs = torch.topk(scores, k=min(k, scores.numel()))

    return [
        {"token": valid_tokens[i], "score": vals[j].item(), "token_index": i}
        for j, i in enumerate(idxs.tolist())
    ]


def topk_patches_for_token(
    patch_proj: torch.Tensor,
    text_token_proj: torch.Tensor,
    tokens: List[str],
    token_index: int,
    attention_mask: torch.Tensor | None = None,
    k: int = 10,
    normalize: bool = False,
):
    """
    returns top-k patch locations for one token
    """
    alignment_map, valid_tokens = compute_alignment_map(
        patch_proj=patch_proj,
        text_token_proj=text_token_proj,
        tokens=tokens,
        attention_mask=attention_mask,
        normalize=normalize,
    )

    if token_index < 0 or token_index >= len(valid_tokens):
        raise IndexError(f"token_index {token_index} out of range for {len(valid_tokens)} valid tokens")

    heatmap = alignment_map[:, :, token_index]  # [H, W]
    h, w = heatmap.shape

    flat = heatmap.reshape(-1)
    vals, idxs = torch.topk(flat, k=min(k, flat.numel()))

    out = []
    for j, flat_idx in enumerate(idxs.tolist()):
        r = flat_idx // w
        c = flat_idx % w
        out.append({
            "row": r,
            "col": c,
            "score": vals[j].item(),
            "token": valid_tokens[token_index],
            "token_index": token_index,
        })
    return out