from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from data_loader import CXRClipInferenceDataset
from calibration.calibration import (
    add_calibrated_predictions_to_df,
    evaluate_predictions_from_df,
    fit_thresholds_from_results_df,
    summarize_thresholds,
)
from models.cxr_clip.load import (
    encode_image,
    encode_text,
    load_cxr_clip,
    score_prompts,
)
from preprocessors.cxr_clip import load_cxr_clip_preprocessor
from utils.prompts import PROMPTS, TARGETS



@torch.no_grad()
def build_text_embedding_cache(pipeline: Dict[str, Any]) -> Dict[str, Dict[str, torch.Tensor]]:
    cache = {}
    device = next(pipeline["model"].parameters()).device

    for label in pipeline["prompts"].keys():
        text_embs = get_text_embeddings_for_label(pipeline, label)

        cache[label] = {
            "pos": text_embs["pos"].to(device),
            "neg": text_embs["neg"].to(device),
        }

    return cache


def init_cxr_clip_pipeline(
    checkpoint_path: Union[str, Path],
    repo_root: Union[str, Path] = "models/cxr_clip/cxr-clip",
    device: Optional[str] = None,
    split: str = "test",
    prompts: Optional[Dict[str, Dict[str, List[str]]]] = None,
    threshold: float = 0.0,
) -> Dict[str, Any]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer, ckpt_config = load_cxr_clip(
        checkpoint_path=checkpoint_path,
        repo_root=repo_root,
        device=device,
    )

    preprocess = load_cxr_clip_preprocessor(
        checkpoint_path=checkpoint_path,
        repo_root=repo_root,
        split=split,
    )

    text_embedding_cache = build_text_embedding_cache({
    "model": model,
    "tokenizer": tokenizer,
    "prompts": prompts if prompts is not None else PROMPTS,
    })

    return {
        "model": model,
        "tokenizer": tokenizer,
        "ckpt_config": ckpt_config,
        "preprocess": preprocess,
        "prompts": prompts if prompts is not None else PROMPTS,
        "text_embedding_cache": text_embedding_cache,
        "targets": list(TARGETS),
        "threshold": threshold,
        "device": device,
        "checkpoint_path": str(checkpoint_path),
        "repo_root": str(repo_root),
    }


@torch.no_grad()
def score_image_tensor(
    pipeline: Dict[str, Any],
    image_tensor: torch.Tensor,
) -> Dict[str, Dict[str, float]]:
    model = pipeline["model"]
    tokenizer = pipeline["tokenizer"]
    prompts = pipeline["prompts"]
    threshold = pipeline["threshold"]

    results: Dict[str, Dict[str, float]] = {}

    for label, prompt_pair in prompts.items():
        pos_scores = score_prompts(
            model=model,
            tokenizer=tokenizer,
            image_tensor=image_tensor,
            prompts=prompt_pair["pos"],
        )
        neg_scores = score_prompts(
            model=model,
            tokenizer=tokenizer,
            image_tensor=image_tensor,
            prompts=prompt_pair["neg"],
        )

        pos_score = float(pos_scores.mean().item())
        neg_score = float(neg_scores.mean().item())
        margin = pos_score - neg_score

        results[label] = {
            "pos_score": pos_score,
            "neg_score": neg_score,
            "margin": margin,
            "pred": float(margin > threshold),
        }

    return results


def preprocess_pil(
    pipeline: Dict[str, Any],
    image: Image.Image,
) -> torch.Tensor:
    preprocess = pipeline["preprocess"]
    return preprocess(image.convert("RGB"))


def preprocess_path(
    pipeline: Dict[str, Any],
    image_path: Union[str, Path],
) -> torch.Tensor:
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with Image.open(image_path) as img:
        return preprocess_pil(pipeline, img)


def predict_pil(
    pipeline: Dict[str, Any],
    image: Image.Image,
) -> Dict[str, Dict[str, float]]:
    image_tensor = preprocess_pil(pipeline, image)
    return score_image_tensor(pipeline, image_tensor)


def predict_path(
    pipeline: Dict[str, Any],
    image_path: Union[str, Path],
) -> Dict[str, Dict[str, float]]:
    image_tensor = preprocess_path(pipeline, image_path)
    return score_image_tensor(pipeline, image_tensor)


def predict_sample(
    pipeline: Dict[str, Any],
    sample: Dict[str, Any],
) -> Dict[str, Any]:
    if "image" not in sample:
        raise KeyError("sample must contain key 'image'")

    results = score_image_tensor(pipeline, sample["image"])

    output = {
        "image_path": sample.get("image_path"),
        "study_id": sample.get("study_id"),
        "subject_id": sample.get("subject_id"),
        "dicom_id": sample.get("dicom_id"),
        "results": results,
    }

    if "labels" in sample:
        labels_tensor = sample["labels"]
        targets = pipeline["targets"]

        if isinstance(labels_tensor, torch.Tensor):
            output["true_labels"] = {
                label: float(labels_tensor[i].item())
                for i, label in enumerate(targets)
            }

    return output


def predict_batch(
    pipeline: Dict[str, Any],
    batch: Dict[str, Any],
) -> List[Dict[str, Any]]:
    if "image" not in batch:
        raise KeyError("batch must contain key 'image'")

    images = batch["image"]
    if not isinstance(images, torch.Tensor):
        raise TypeError("batch['image'] must be a torch.Tensor")

    batch_size = images.shape[0]
    outputs: List[Dict[str, Any]] = []

    image_paths = batch.get("image_path", [None] * batch_size)
    study_ids = batch.get("study_id", [None] * batch_size)
    subject_ids = batch.get("subject_id", [None] * batch_size)
    dicom_ids = batch.get("dicom_id", [None] * batch_size)

    for i in range(batch_size):
        sample = {
            "image": images[i],
            "image_path": image_paths[i],
            "study_id": study_ids[i],
            "subject_id": subject_ids[i],
            "dicom_id": dicom_ids[i],
        }

        if "labels" in batch:
            sample["labels"] = batch["labels"][i]

        outputs.append(predict_sample(pipeline, sample))

    return outputs


def get_positive_predictions(
    results: Dict[str, Dict[str, float]],
    threshold: float = 0.0,
) -> Dict[str, Dict[str, float]]:
    return {
        label: info
        for label, info in results.items()
        if info["margin"] > threshold
    }


def top_k_predictions(
    results: Dict[str, Dict[str, float]],
    k: int = 3,
) -> List[tuple[str, Dict[str, float]]]:
    return sorted(
        results.items(),
        key=lambda x: x[1]["margin"],
        reverse=True,
    )[:k]


@torch.no_grad()
def get_image_embedding(
    pipeline: Dict[str, Any],
    image_tensor: torch.Tensor,
) -> torch.Tensor:
    model = pipeline["model"]
    return encode_image(model, image_tensor)


@torch.no_grad()
def get_text_embeddings_for_label(
    pipeline: Dict[str, Any],
    label: str,
) -> Dict[str, torch.Tensor]:
    model = pipeline["model"]
    tokenizer = pipeline["tokenizer"]
    prompts = pipeline["prompts"][label]

    pos_emb = encode_text(model, tokenizer, prompts["pos"])
    neg_emb = encode_text(model, tokenizer, prompts["neg"])

    return {
        "pos": pos_emb,
        "neg": neg_emb,
    }


@torch.no_grad()
def score_image_embedding(
    pipeline: Dict[str, Any],
    image_embedding: Union[np.ndarray, torch.Tensor],
) -> Dict[str, Dict[str, float]]:
    """
    Score a precomputed image embedding against all prompt pairs.
    """
    prompts = pipeline["prompts"]
    threshold = pipeline["threshold"]
    device = next(pipeline["model"].parameters()).device

    if isinstance(image_embedding, np.ndarray):
        image_embedding = torch.from_numpy(image_embedding)

    if image_embedding.ndim == 1:
        image_embedding = image_embedding.unsqueeze(0)

    image_embedding = image_embedding.float().to(device)
    image_embedding = image_embedding / torch.norm(image_embedding, dim=1, keepdim=True)

    results: Dict[str, Dict[str, float]] = {}

    text_cache = pipeline["text_embedding_cache"]

    for label in prompts.keys():
        pos_emb = text_cache[label]["pos"].to(device)
        neg_emb = text_cache[label]["neg"].to(device)

        pos_score = float((image_embedding @ pos_emb.T).mean().item())
        neg_score = float((image_embedding @ neg_emb.T).mean().item())
        margin = pos_score - neg_score

        results[label] = {
            "pos_score": pos_score,
            "neg_score": neg_score,
            "margin": margin,
            "pred": float(margin > threshold),
        }

    return results


@torch.no_grad()
def extract_dataset_image_embeddings(
    pipeline: Dict[str, Any],
    dataset,
    batch_size: int = 16,
    num_workers: int = 0,
) -> Dict[str, Any]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    all_embeddings: List[np.ndarray] = []
    metadata_rows: List[Dict[str, Any]] = []

    for batch in loader:
        image_batch = batch["image"]
        emb = get_image_embedding(pipeline, image_batch)
        emb = emb.detach().cpu().numpy()
        all_embeddings.append(emb)

        batch_size_actual = emb.shape[0]

        image_paths = batch.get("image_path", [None] * batch_size_actual)
        study_ids = batch.get("study_id", [None] * batch_size_actual)
        subject_ids = batch.get("subject_id", [None] * batch_size_actual)
        dicom_ids = batch.get("dicom_id", [None] * batch_size_actual)

        labels_tensor = batch.get("labels", None)

        for i in range(batch_size_actual):
            row = {
                "image_path": image_paths[i],
                "study_id": study_ids[i],
                "subject_id": subject_ids[i],
                "dicom_id": dicom_ids[i],
            }

            if labels_tensor is not None:
                for j, label in enumerate(pipeline["targets"]):
                    row[f"{label}_true"] = float(labels_tensor[i, j].item())

            metadata_rows.append(row)

    embeddings = np.concatenate(all_embeddings, axis=0)
    metadata_df = pd.DataFrame(metadata_rows)

    return {
        "embeddings": embeddings,
        "metadata": metadata_df,
    }


def save_dataset_image_embeddings(
    pipeline: Dict[str, Any],
    dataset,
    output_prefix: Union[str, Path],
    batch_size: int = 16,
    num_workers: int = 0,
) -> Dict[str, Any]:
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    extracted = extract_dataset_image_embeddings(
        pipeline=pipeline,
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    embeddings = extracted["embeddings"]
    metadata_df = extracted["metadata"]

    embeddings_path = str(output_prefix) + "_embeddings.npy"
    metadata_path = str(output_prefix) + "_metadata.csv"

    np.save(embeddings_path, embeddings)
    metadata_df.to_csv(metadata_path, index=False)

    return {
        "embeddings_path": embeddings_path,
        "metadata_path": metadata_path,
        "num_samples": len(metadata_df),
        "embedding_dim": embeddings.shape[1],
    }


def load_saved_dataset_image_embeddings(
    output_prefix: Union[str, Path],
) -> Dict[str, Any]:
    output_prefix = Path(output_prefix)
    embeddings_path = str(output_prefix) + "_embeddings.npy"
    metadata_path = str(output_prefix) + "_metadata.csv"

    if not Path(embeddings_path).exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
    if not Path(metadata_path).exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    embeddings = np.load(embeddings_path)
    metadata = pd.read_csv(metadata_path)

    return {
        "embeddings": embeddings,
        "metadata": metadata,
        "embeddings_path": embeddings_path,
        "metadata_path": metadata_path,
    }


def get_or_create_dataset_image_embeddings(
    pipeline: Dict[str, Any],
    dataset,
    output_prefix: Union[str, Path],
    batch_size: int = 16,
    num_workers: int = 0,
) -> Dict[str, Any]:
    """
    If saved embeddings exist, load them.
    Otherwise extract image embeddings, save them, then load them back.
    """
    output_prefix = Path(output_prefix)
    embeddings_path = str(output_prefix) + "_embeddings.npy"
    metadata_path = str(output_prefix) + "_metadata.csv"

    if Path(embeddings_path).exists() and Path(metadata_path).exists():
        loaded = load_saved_dataset_image_embeddings(output_prefix)
        loaded["source"] = "loaded"
        return loaded

    save_info = save_dataset_image_embeddings(
        pipeline=pipeline,
        dataset=dataset,
        output_prefix=output_prefix,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    loaded = load_saved_dataset_image_embeddings(output_prefix)
    loaded["source"] = "created"
    loaded["save_info"] = save_info
    return loaded


def collect_results_from_saved_embeddings(
    pipeline: Dict[str, Any],
    embeddings: np.ndarray,
    metadata_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the same style of results dataframe as collect_results_from_dataset(...),
    but from saved image embeddings instead of rerunning image encoding.
    """
    rows: List[Dict[str, Any]] = []

    for i in range(len(metadata_df)):
        emb = embeddings[i]
        results = score_image_embedding(pipeline, emb)

        meta = metadata_df.iloc[i]
        row: Dict[str, Any] = {
            "image_path": meta.get("image_path"),
            "study_id": meta.get("study_id"),
            "subject_id": meta.get("subject_id"),
            "dicom_id": meta.get("dicom_id"),
        }

        for label in pipeline["targets"]:
            true_col = f"{label}_true"
            row[true_col] = int(meta[true_col]) if true_col in meta.index else 0
            row[f"{label}_margin"] = float(results[label]["margin"])
            row[f"{label}_pos"] = float(results[label]["pos_score"])
            row[f"{label}_neg"] = float(results[label]["neg_score"])

        rows.append(row)

    return pd.DataFrame(rows)


def collect_results_from_dataset(
    dataset,
    pipeline: Dict[str, Any],
    predict_sample_fn,
) -> pd.DataFrame:
    """
    Kept here so the notebook can call a single pipeline module.
    """
    rows: List[Dict[str, Any]] = []

    for i in range(len(dataset)):
        sample = dataset[i]
        output = predict_sample_fn(pipeline, sample)

        row: Dict[str, Any] = {
            "image_path": output.get("image_path"),
            "study_id": output.get("study_id"),
            "subject_id": output.get("subject_id"),
            "dicom_id": output.get("dicom_id"),
        }

        for label in pipeline["targets"]:
            row[f"{label}_true"] = int(output["true_labels"][label])
            row[f"{label}_margin"] = float(output["results"][label]["margin"])
            row[f"{label}_pos"] = float(output["results"][label]["pos_score"])
            row[f"{label}_neg"] = float(output["results"][label]["neg_score"])

        rows.append(row)

    return pd.DataFrame(rows)


def get_or_collect_results_df(
    pipeline: Dict[str, Any],
    dataset,
    output_prefix: Union[str, Path],
    batch_size: int = 16,
    num_workers: int = 0,
    use_saved_embeddings: bool = True,
) -> Dict[str, Any]:
    """
    If use_saved_embeddings=True:
      - load embeddings if they already exist
      - otherwise extract/save embeddings
      - then compute raw results from embeddings

    If use_saved_embeddings=False:
      - rerun full inference directly from dataset
    """
    if use_saved_embeddings:
        emb_pack = get_or_create_dataset_image_embeddings(
            pipeline=pipeline,
            dataset=dataset,
            output_prefix=output_prefix,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        results_df = collect_results_from_saved_embeddings(
            pipeline=pipeline,
            embeddings=emb_pack["embeddings"],
            metadata_df=emb_pack["metadata"],
        )

        return {
            "results_df": results_df,
            "embedding_info": emb_pack,
            "source": emb_pack.get("source", "loaded"),
        }

    results_df = collect_results_from_dataset(
        dataset=dataset,
        pipeline=pipeline,
        predict_sample_fn=predict_sample,
    )
    return {
        "results_df": results_df,
        "embedding_info": None,
        "source": "direct_inference",
    }


def run_cxr_clip_calibration_and_evaluation(
    pipeline: Dict[str, Any],
    csv_path: Union[str, Path],
    output_dir: Union[str, Path] = "calibration",
    test_size: float = 0.2,
    random_state: int = 42,
    use_saved_embeddings: bool = True,
    batch_size: int = 16,
    num_workers: int = 0,
    threshold_objective: str = "f1",
) -> Dict[str, Any]:
    """
    Full workflow:
      - load csv
      - split by subject_id
      - create datasets
      - if embeddings exist, use them
      - otherwise run inference/extract embeddings and save them
      - fit thresholds on calibration split
      - evaluate on held-out test split
      - save outputs
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    subjects = df["subject_id"].unique()
    calibration_subjects, test_subjects = train_test_split(
        subjects,
        test_size=test_size,
        random_state=random_state,
    )

    calibration_df = df[df["subject_id"].isin(calibration_subjects)].reset_index(drop=True)
    test_df = df[df["subject_id"].isin(test_subjects)].reset_index(drop=True)

    calibration_dataset = CXRClipInferenceDataset(
        calibration_df,
        transform=pipeline["preprocess"],
    )
    test_dataset = CXRClipInferenceDataset(
        test_df,
        transform=pipeline["preprocess"],
    )

    calibration_pack = get_or_collect_results_df(
        pipeline=pipeline,
        dataset=calibration_dataset,
        output_prefix=output_dir / "calibration",
        batch_size=batch_size,
        num_workers=num_workers,
        use_saved_embeddings=use_saved_embeddings,
    )
    cal_results_df = calibration_pack["results_df"]

    thresholds = fit_thresholds_from_results_df(
        cal_results_df,
        objective=threshold_objective,
    )
    threshold_table = summarize_thresholds(thresholds)

    cal_results_cal = add_calibrated_predictions_to_df(cal_results_df, thresholds)
    cal_metrics_df = evaluate_predictions_from_df(cal_results_cal)

    test_pack = get_or_collect_results_df(
        pipeline=pipeline,
        dataset=test_dataset,
        output_prefix=output_dir / "test",
        batch_size=batch_size,
        num_workers=num_workers,
        use_saved_embeddings=use_saved_embeddings,
    )
    test_results_df = test_pack["results_df"]

    test_results_cal = add_calibrated_predictions_to_df(test_results_df, thresholds)
    test_metrics_df = evaluate_predictions_from_df(test_results_cal)

    threshold_table.to_csv(output_dir / "thresholds_calibration.csv", index=False)

    cal_results_df.to_csv(output_dir / "calibration_raw_results.csv", index=False)
    cal_results_cal.to_csv(output_dir / "calibration_calibrated_results.csv", index=False)
    cal_metrics_df.to_csv(output_dir / "calibration_metrics.csv", index=False)

    test_results_df.to_csv(output_dir / "test_raw_results.csv", index=False)
    test_results_cal.to_csv(output_dir / "test_calibrated_results.csv", index=False)
    test_metrics_df.to_csv(output_dir / "test_metrics.csv", index=False)

    return {
        "full_df": df,
        "calibration_df": calibration_df,
        "test_df": test_df,
        "calibration_dataset": calibration_dataset,
        "test_dataset": test_dataset,
        "thresholds": thresholds,
        "threshold_table": threshold_table,
        "cal_results_df": cal_results_df,
        "cal_results_cal": cal_results_cal,
        "cal_metrics_df": cal_metrics_df,
        "test_results_df": test_results_df,
        "test_results_cal": test_results_cal,
        "test_metrics_df": test_metrics_df,
        "calibration_embedding_info": calibration_pack["embedding_info"],
        "test_embedding_info": test_pack["embedding_info"],
        "calibration_result_source": calibration_pack["source"],
        "test_result_source": test_pack["source"],
    }