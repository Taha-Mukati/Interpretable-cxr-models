from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

import random
import torch
import torch.nn.functional as F



def train_linear_cav(
    positive_embeddings: torch.Tensor,
    negative_embeddings: torch.Tensor,
    lr: float = 1e-2,
    weight_decay: float = 1e-4,
    steps: int = 500,
):
    """
    Learn a linear separator; use its weight vector as the CAV.
    """
    x = torch.cat([positive_embeddings, negative_embeddings], dim=0)
    y = torch.cat([
        torch.ones(len(positive_embeddings)),
        torch.zeros(len(negative_embeddings)),
    ], dim=0).unsqueeze(1)

    d = x.shape[1]
    w = torch.zeros(d, 1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    optimizer = torch.optim.Adam([w, b], lr=lr, weight_decay=weight_decay)

    for _ in range(steps):
        logits = x @ w + b
        loss = F.binary_cross_entropy_with_logits(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    cav = w.detach().squeeze(1)
    cav = F.normalize(cav, dim=0)
    return cav


def build_cav_from_samples(
    positives: Sequence[Dict[str, Any]],
    negatives: Sequence[Dict[str, Any]],
    embedding_key: str = "image_emb",
):
    pos = torch.stack([s[embedding_key] for s in positives], dim=0)
    neg = torch.stack([s[embedding_key] for s in negatives], dim=0)
    return train_linear_cav(pos, neg)


def image_text_similarity_score(image_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
    image_emb = F.normalize(image_emb, dim=-1)
    text_emb = F.normalize(text_emb, dim=-1)
    return torch.sum(image_emb * text_emb, dim=-1)


def tcav_score_from_embeddings(
    samples: Sequence[Dict[str, Any]],
    cav: torch.Tensor,
    embedding_key: str = "image_emb",
):
    """
    Simple embedding-space TCAV proxy:
    proportion of samples whose embedding projects positively onto the CAV.
    """
    embs = torch.stack([s[embedding_key] for s in samples], dim=0)
    embs = F.normalize(embs, dim=-1)
    directional = embs @ cav
    return float((directional > 0).float().mean().item()), directional


def similarity_tcav_score(
    samples: Sequence[Dict[str, Any]],
    cav: torch.Tensor,
    epsilon: float = 1e-2,
    embedding_key: str = "image_emb",
):
    """
    TCAV-like score using finite differences on image-text similarity.

    For each sample:
      score_plus  = sim(e + eps*cav, text)
      score_minus = sim(e - eps*cav, text)
      directional derivative sign ~ score_plus - score_minus
    """
    signs = []

    for s in samples:
        e = F.normalize(s[embedding_key].float(), dim=0)
        t = F.normalize(s["text_emb"].float(), dim=0)

        e_plus = F.normalize(e + epsilon * cav, dim=0)
        e_minus = F.normalize(e - epsilon * cav, dim=0)

        score_plus = torch.dot(e_plus, t)
        score_minus = torch.dot(e_minus, t)

        signs.append(float((score_plus - score_minus).item()))

    signs_t = torch.tensor(signs)
    tcav = float((signs_t > 0).float().mean().item())
    return tcav, signs_t


def retrieve_prototypes(
    query_embedding: torch.Tensor,
    samples: Sequence[Dict[str, Any]],
    embedding_key: str = "image_emb",
    k: int = 5,
):
    query_embedding = F.normalize(query_embedding.float(), dim=0)

    scored = []
    for s in samples:
        emb = F.normalize(s[embedding_key].float(), dim=0)
        sim = torch.dot(query_embedding, emb).item()
        scored.append((sim, s))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:k]