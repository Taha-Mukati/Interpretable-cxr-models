from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
import torch.nn.functional as F







def retrieve_prototypes(
    query_embedding: torch.Tensor,
    samples: Sequence[Dict[str, Any]],
    embedding_key: str = "image_emb",
    k: int = 5,
    exclude_same_image: bool = True,
    query_image_path: str | None = None,
):
    """
    Retrieve top-k nearest neighbours in embedding space.

    Args:
        query_embedding: [D]
        samples: list of sample dicts
        embedding_key: which embedding to compare against
        k: number of neighbours
        exclude_same_image: exclude the exact same image if present
        query_image_path: path of the query image for exclusion check

    Returns:
        list of dicts with similarity score and sample
    """
    query_embedding = F.normalize(query_embedding.float(), dim=0)

    scored = []
    for s in samples:
        emb = s.get(embedding_key)
        if emb is None:
            continue

        if exclude_same_image and query_image_path is not None:
            meta = s.get("metadata", {})
            if meta.get("image_path") == query_image_path:
                continue

        emb = F.normalize(emb.float(), dim=0)
        sim = torch.dot(query_embedding, emb).item()

        scored.append({
            "similarity": sim,
            "sample": s,
        })

    scored.sort(key=lambda x: x["similarity"], reverse=True)
    return scored[:k]

def retrieve_reports(
    query_embedding: torch.Tensor,
    samples: Sequence[Dict[str, Any]],
    embedding_key: str = "text_emb",
    k: int = 5,
    exclude_same_report: bool = True,
    query_report_text: str | None = None,
):
    """
    Retrieve top-k nearest reports in embedding space.
    """
    query_embedding = F.normalize(query_embedding.float(), dim=0)

    scored = []
    for s in samples:
        emb = s.get(embedding_key)
        if emb is None:
            continue

        report_text = (
            s["metadata"].get("report_text")
            or s["metadata"].get("current_report_text")
            or ""
        )

        if exclude_same_report and query_report_text is not None:
            if report_text == query_report_text:
                continue

        emb = F.normalize(emb.float(), dim=0)
        sim = torch.dot(query_embedding, emb).item()

        scored.append({
            "similarity": sim,
            "sample": s,
        })

    scored.sort(key=lambda x: x["similarity"], reverse=True)
    return scored[:k]