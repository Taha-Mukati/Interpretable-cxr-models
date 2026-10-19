from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ABNORMAL_LABELS = [
    "Pneumothorax",
    "Pleural Effusion",
    "Support Devices",
    "Consolidation",
    "Cardiomegaly",
    "Atelectasis",
    "Edema",
]

ALL_LABELS = ABNORMAL_LABELS + ["No Finding"]


def compute_confusion_counts(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, int]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def precision_recall_f1_from_counts(counts: Dict[str, int]) -> Dict[str, float]:
    tp, fp, tn, fn = counts["tp"], counts["fp"], counts["tn"], counts["fn"]

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    balanced_accuracy = (recall + specificity) / 2.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "f1": float(f1),
    }


def compute_binary_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
    counts = compute_confusion_counts(y_true, y_pred)
    metrics = precision_recall_f1_from_counts(counts)
    return {**counts, **metrics}


def _safe_score_range(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)

    if scores.size == 0:
        return np.array([0.0], dtype=float)

    smin = float(scores.min())
    smax = float(scores.max())

    if np.isclose(smin, smax):
        return np.array([smin], dtype=float)

    return np.linspace(smin, smax, 400)


def find_best_threshold(
    y_true: Sequence[int],
    scores: Sequence[float],
    objective: str = "f1",
    num_steps: int = 400,
) -> Dict[str, float]:
    """
    Find the best threshold for one label from continuous scores.

    Parameters
    ----------
    y_true:
        Binary ground-truth labels.
    scores:
        Continuous scores, e.g. CXR-CLIP margins.
    objective:
        One of: 'f1', 'balanced_accuracy', 'precision', 'recall', 'accuracy'
    num_steps:
        Number of thresholds to scan if the score range is non-constant.
    """
    valid_objectives = {"f1", "balanced_accuracy", "precision", "recall", "accuracy"}
    if objective not in valid_objectives:
        raise ValueError(f"objective must be one of {sorted(valid_objectives)}")

    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)

    if y_true.shape[0] != scores.shape[0]:
        raise ValueError("y_true and scores must have the same length")

    thresholds = _safe_score_range(scores)
    if thresholds.size > 1 and num_steps != len(thresholds):
        thresholds = np.linspace(float(scores.min()), float(scores.max()), num_steps)

    best = {
        "threshold": 0.0,
        "objective_value": -1.0,
        "f1": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "specificity": 0.0,
        "accuracy": 0.0,
        "balanced_accuracy": 0.0,
    }

    for thr in thresholds:
        y_pred = (scores > thr).astype(int)
        metrics = compute_binary_metrics(y_true, y_pred)

        score = float(metrics[objective])
        if score > best["objective_value"]:
            best = {
                "threshold": float(thr),
                "objective_value": score,
                "f1": float(metrics["f1"]),
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "specificity": float(metrics["specificity"]),
                "accuracy": float(metrics["accuracy"]),
                "balanced_accuracy": float(metrics["balanced_accuracy"]),
            }

    return best


def fit_thresholds_from_results_df(
    results_df: pd.DataFrame,
    abnormal_labels: Optional[Sequence[str]] = None,
    objective: str = "f1",
    margin_suffix: str = "_margin",
    true_suffix: str = "_true",
) -> Dict[str, Dict[str, float]]:
    """
    Fit one threshold per abnormal class from a dataframe containing:
      <label>_true
      <label>_margin
    """
    labels = list(abnormal_labels) if abnormal_labels is not None else list(ABNORMAL_LABELS)

    thresholds: Dict[str, Dict[str, float]] = {}
    for label in labels:
        y_true_col = f"{label}{true_suffix}"
        score_col = f"{label}{margin_suffix}"

        if y_true_col not in results_df.columns:
            raise KeyError(f"Missing column: {y_true_col}")
        if score_col not in results_df.columns:
            raise KeyError(f"Missing column: {score_col}")

        best = find_best_threshold(
            y_true=results_df[y_true_col].values,
            scores=results_df[score_col].values,
            objective=objective,
        )
        thresholds[label] = best

    return thresholds


def apply_calibrated_thresholds_to_results(
    results: Dict[str, Dict[str, float]],
    thresholds: Dict[str, Dict[str, float]],
    abnormal_labels: Optional[Sequence[str]] = None,
) -> Dict[str, int]:
    """
    Apply per-class thresholds to raw per-label results from predict_sample(...).

    Parameters
    ----------
    results:
        Dict like output["results"] from your CXR-CLIP pipeline.
    thresholds:
        Dict[label] -> {"threshold": ...}
    abnormal_labels:
        Labels to threshold directly. "No Finding" is derived automatically.

    Returns
    -------
    preds:
        Binary predictions for all labels, including derived "No Finding".
    """
    labels = list(abnormal_labels) if abnormal_labels is not None else list(ABNORMAL_LABELS)

    preds: Dict[str, int] = {}
    for label in labels:
        if label not in results:
            raise KeyError(f"Missing results for label: {label}")
        if label not in thresholds:
            raise KeyError(f"Missing threshold for label: {label}")

        thr = float(thresholds[label]["threshold"])
        margin = float(results[label]["margin"])
        preds[label] = int(margin > thr)

    preds["No Finding"] = int(sum(preds[label] for label in labels) == 0)
    return preds


def apply_calibrated_thresholds_to_row(
    row: pd.Series,
    thresholds: Dict[str, Dict[str, float]],
    abnormal_labels: Optional[Sequence[str]] = None,
    margin_suffix: str = "_margin",
) -> Dict[str, int]:
    """
    Apply thresholds to a dataframe row containing <label>_margin columns.
    """
    labels = list(abnormal_labels) if abnormal_labels is not None else list(ABNORMAL_LABELS)

    preds: Dict[str, int] = {}
    for label in labels:
        score_col = f"{label}{margin_suffix}"
        if score_col not in row.index:
            raise KeyError(f"Missing column: {score_col}")
        if label not in thresholds:
            raise KeyError(f"Missing threshold for label: {label}")

        thr = float(thresholds[label]["threshold"])
        preds[label] = int(float(row[score_col]) > thr)

    preds["No Finding"] = int(sum(preds[label] for label in labels) == 0)
    return preds


def add_calibrated_predictions_to_df(
    results_df: pd.DataFrame,
    thresholds: Dict[str, Dict[str, float]],
    abnormal_labels: Optional[Sequence[str]] = None,
    margin_suffix: str = "_margin",
    pred_suffix: str = "_pred_calibrated",
) -> pd.DataFrame:
    """
    Return a copy of results_df with calibrated prediction columns added.
    """
    labels = list(abnormal_labels) if abnormal_labels is not None else list(ABNORMAL_LABELS)
    df = results_df.copy()

    pred_rows: List[Dict[str, int]] = []
    for _, row in df.iterrows():
        preds = apply_calibrated_thresholds_to_row(
            row=row,
            thresholds=thresholds,
            abnormal_labels=labels,
            margin_suffix=margin_suffix,
        )
        pred_rows.append(preds)

    pred_df = pd.DataFrame(pred_rows, index=df.index)
    for label in labels + ["No Finding"]:
        df[f"{label}{pred_suffix}"] = pred_df[label].astype(int)

    return df


def evaluate_predictions_from_df(
    results_df: pd.DataFrame,
    labels: Optional[Sequence[str]] = None,
    true_suffix: str = "_true",
    pred_suffix: str = "_pred_calibrated",
) -> pd.DataFrame:
    """
    Evaluate per-label metrics from a dataframe containing true/pred columns.
    """
    eval_labels = list(labels) if labels is not None else list(ALL_LABELS)

    rows: List[Dict[str, Any]] = []
    for label in eval_labels:
        true_col = f"{label}{true_suffix}"
        pred_col = f"{label}{pred_suffix}"

        if true_col not in results_df.columns:
            raise KeyError(f"Missing column: {true_col}")
        if pred_col not in results_df.columns:
            raise KeyError(f"Missing column: {pred_col}")

        metrics = compute_binary_metrics(
            y_true=results_df[true_col].values,
            y_pred=results_df[pred_col].values,
        )
        rows.append({"label": label, **metrics})

    return pd.DataFrame(rows)


def collect_results_from_dataset(
    dataset: Any,
    pipeline: Dict[str, Any],
    predict_sample_fn: Any,
    labels: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Run inference over a dataset and collect raw margins + true labels into a dataframe.

    Parameters
    ----------
    dataset:
        Your CXRClipInferenceDataset instance.
    pipeline:
        Initialized pipeline object/dict.
    predict_sample_fn:
        Function like predict_sample(pipeline, sample).
    labels:
        Labels to collect. Defaults to ALL_LABELS.
    """
    use_labels = list(labels) if labels is not None else list(ALL_LABELS)

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

        for label in use_labels:
            row[f"{label}_true"] = int(output["true_labels"][label])
            row[f"{label}_margin"] = float(output["results"][label]["margin"])
            row[f"{label}_pos"] = float(output["results"][label]["pos_score"])
            row[f"{label}_neg"] = float(output["results"][label]["neg_score"])

        rows.append(row)

    return pd.DataFrame(rows)


def summarize_thresholds(thresholds: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    rows = []
    for label, info in thresholds.items():
        rows.append(
            {
                "label": label,
                "threshold": float(info["threshold"]),
                "objective_value": float(info["objective_value"]),
                "f1": float(info["f1"]),
                "precision": float(info["precision"]),
                "recall": float(info["recall"]),
                "specificity": float(info["specificity"]),
                "accuracy": float(info["accuracy"]),
                "balanced_accuracy": float(info["balanced_accuracy"]),
            }
        )
    return pd.DataFrame(rows).sort_values("label").reset_index(drop=True)