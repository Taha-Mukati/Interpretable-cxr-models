from __future__ import annotations

import random
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Tuple
import gc

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data_loader import SingleInferenceDataset, TemporalDataset
from preprocessors.biovil_t import load_biovil_preprocessor
from collate.biovil_t import collate_biovil_single, collate_biovil_temporal
from models.biovil_t.load import load_biovil_t

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def ensure_output_dirs(root: str = "embeddings/biovil_t") -> Dict[str, Path]:
    root_path = Path(root)
    single_dir = root_path / "single"
    temporal_dir = root_path / "temporal"

    single_chunks = single_dir / "chunks"
    temporal_chunks = temporal_dir / "chunks"

    root_path.mkdir(parents=True, exist_ok=True)
    single_dir.mkdir(parents=True, exist_ok=True)
    temporal_dir.mkdir(parents=True, exist_ok=True)
    single_chunks.mkdir(parents=True, exist_ok=True)
    temporal_chunks.mkdir(parents=True, exist_ok=True)

    return {
        "root": root_path,
        "single": single_dir,
        "temporal": temporal_dir,
        "single_chunks": single_chunks,
        "temporal_chunks": temporal_chunks,
    }


def build_biovil_datasets(
    single_csv_path: str,
    temporal_csv_path: str,
) -> Tuple[SingleInferenceDataset, TemporalDataset]:
    single_ds = SingleInferenceDataset(single_csv_path)
    temporal_ds = TemporalDataset(temporal_csv_path)
    return single_ds, temporal_ds


def build_biovil_preprocessors(
    model_name: str = "microsoft/BiomedVLP-BioViL-T",
):
    image_processor = load_biovil_preprocessor(model_name=model_name, mode="image")
    tokenizer = load_biovil_preprocessor(model_name=model_name, mode="text")
    return image_processor, tokenizer


def build_single_loader(
    single_ds: SingleInferenceDataset,
    image_processor,
    tokenizer,
    single_batch_size: int = 8,
    num_workers: int = 0,
):
    return DataLoader(
        single_ds,
        batch_size=single_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=partial(
            collate_biovil_single,
            image_processor=image_processor,
            tokenizer=tokenizer,
        ),
    )


def build_temporal_loader(
    temporal_ds: TemporalDataset,
    image_processor,
    tokenizer,
    temporal_batch_size: int = 4,
    num_workers: int = 0,
):
    return DataLoader(
        temporal_ds,
        batch_size=temporal_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=partial(
            collate_biovil_temporal,
            image_processor=image_processor,
            tokenizer=tokenizer,
        ),
    )


def load_biovil_components(
    model_name: str = "microsoft/BiomedVLP-BioViL-T",
    device: str | None = None,
):
    if device is None:
        device = get_device()

    image_processor, tokenizer = build_biovil_preprocessors(model_name=model_name)
    model = load_biovil_t(model_name=model_name, device=device)

    return {
        "device": device,
        "image_processor": image_processor,
        "tokenizer": tokenizer,
        "model": model,
    }


def _tokenize_batch_to_strings(tokenizer, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> List[List[str]]:
    tokens_batch: List[List[str]] = []
    for ids, mask in zip(input_ids, attention_mask):
        valid_ids = ids[mask.bool()].tolist()
        tokens_batch.append(tokenizer.convert_ids_to_tokens(valid_ids))
    return tokens_batch


def _cpu_clone_tuple_tensors(xs) -> List[torch.Tensor] | None:
    if xs is None:
        return None
    return [x.detach().cpu().clone() for x in xs]


def _cleanup_after_batch(device: str) -> None:
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()


def save_single_chunk(chunk_samples: List[Dict[str, Any]], output_dir: str | Path, chunk_idx: int) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = []
    labels_list = []

    image_embs = []
    text_embs = []
    image_text_sims = []
    image_patch_projs = []

    text_token_projs = []
    tokens_list = []
    input_ids_list = []
    attention_masks_list = []
    text_attentions_list = []
    text_hidden_states_list = []

    for sample in chunk_samples:
        metadata.append({
            "subject_id": sample["subject_id"],
            "study_id": sample["study_id"],
            "dicom_id": sample["dicom_id"],
            "image_path": sample["image_path"],
            "report_path": sample["report_path"],
            "report_text": sample["report_text"],
        })

        labels_list.append(sample["labels"])

        image_embs.append(sample["image_emb"])
        text_embs.append(sample["text_emb"])
        image_text_sims.append(sample["image_text_similarity"])
        image_patch_projs.append(sample["image_patch_proj"])

        text_token_projs.append(sample["text_token_proj"])
        tokens_list.append(sample["tokens"])
        input_ids_list.append(sample["input_ids"])
        attention_masks_list.append(sample["attention_mask"])
        text_attentions_list.append(sample["text_attentions"])
        text_hidden_states_list.append(sample["text_hidden_states"])

    chunk_data = {
        "metadata": metadata,
        "labels": labels_list,
        "image_emb": torch.stack(image_embs),
        "text_emb": torch.stack(text_embs),
        "image_text_similarity": torch.tensor(image_text_sims, dtype=torch.float32),
        "image_patch_proj": torch.stack(image_patch_projs),  # [N, P, D]
        "text_token_proj": text_token_projs,
        "tokens": tokens_list,
        "input_ids": input_ids_list,
        "attention_mask": attention_masks_list,
        "text_attentions": text_attentions_list,
        "text_hidden_states": text_hidden_states_list,
    }

    chunk_path = output_dir / f"single_chunk_{chunk_idx:04d}.pt"
    torch.save(chunk_data, chunk_path)
    return chunk_path


def save_temporal_chunk(chunk_samples: List[Dict[str, Any]], output_dir: str | Path, chunk_idx: int) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = []
    labels_list = []

    prior_embs = []
    current_embs = []
    text_embs = []

    prior_text_sims = []
    current_text_sims = []
    prior_current_sims = []

    prior_patch_projs = []
    current_patch_projs = []

    current_patch_features_list = []
    temporal_global_embs = []
    temporal_patch_projs = []

    text_token_projs = []
    tokens_list = []
    input_ids_list = []
    attention_masks_list = []

    temporal_attentions_list = []
    text_attentions_list = []
    text_hidden_states_list = []

    temporal_global_all_present = True

    for sample in chunk_samples:
        metadata.append({
            "subject_id": sample["subject_id"],
            "prior_study_id": sample["prior_study_id"],
            "prior_dicom_id": sample["prior_dicom_id"],
            "current_study_id": sample["current_study_id"],
            "current_dicom_id": sample["current_dicom_id"],
            "gap_days": sample["gap_days"],
            "prior_image_path": sample["prior_image_path"],
            "current_image_path": sample["current_image_path"],
            "report_path": sample["report_path"],
            "current_report_text": sample["current_report_text"],
        })

        labels_list.append(sample["labels"])

        prior_embs.append(sample["prior_emb"])
        current_embs.append(sample["current_emb"])
        text_embs.append(sample["text_emb"])

        prior_text_sims.append(sample["prior_text_similarity"])
        current_text_sims.append(sample["current_text_similarity"])
        prior_current_sims.append(sample["prior_current_similarity"])

        prior_patch_projs.append(sample["prior_patch_proj"])
        current_patch_projs.append(sample["current_patch_proj"])

        current_patch_features_list.append(sample["current_patch_features"])
        temporal_patch_projs.append(sample["temporal_patch_proj"])

        if sample["temporal_global_emb"] is None:
            temporal_global_all_present = False
        else:
            temporal_global_embs.append(sample["temporal_global_emb"])

        text_token_projs.append(sample["text_token_proj"])
        tokens_list.append(sample["tokens"])
        input_ids_list.append(sample["input_ids"])
        attention_masks_list.append(sample["attention_mask"])

        temporal_attentions_list.append(sample["temporal_attentions"])
        text_attentions_list.append(sample["text_attentions"])
        text_hidden_states_list.append(sample["text_hidden_states"])

    chunk_data = {
        "metadata": metadata,
        "labels": labels_list,
        "prior_emb": torch.stack(prior_embs),
        "current_emb": torch.stack(current_embs),
        "text_emb": torch.stack(text_embs),
        "prior_text_similarity": torch.tensor(prior_text_sims, dtype=torch.float32),
        "current_text_similarity": torch.tensor(current_text_sims, dtype=torch.float32),
        "prior_current_similarity": torch.tensor(prior_current_sims, dtype=torch.float32),
        "prior_patch_proj": torch.stack(prior_patch_projs),            # [N, P, D]
        "current_patch_proj": torch.stack(current_patch_projs),        # [N, P, D]
        "current_patch_features": torch.stack(current_patch_features_list),  # [N, C, H, W]
        "temporal_global_emb": (
            torch.stack(temporal_global_embs) if temporal_global_all_present else None
        ),
        "temporal_patch_proj": torch.stack(temporal_patch_projs),      # [N, P, D]
        "text_token_proj": text_token_projs,
        "tokens": tokens_list,
        "input_ids": input_ids_list,
        "attention_mask": attention_masks_list,
        "temporal_attentions": temporal_attentions_list,
        "text_attentions": text_attentions_list,
        "text_hidden_states": text_hidden_states_list,
    }

    chunk_path = output_dir / f"temporal_chunk_{chunk_idx:04d}.pt"
    torch.save(chunk_data, chunk_path)
    return chunk_path


def save_manifest(manifest: Dict[str, Any], manifest_path: str | Path) -> Path:
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(manifest, manifest_path)
    return manifest_path


def extract_single_outputs_chunked(
    model,
    loader,
    tokenizer,
    device: str,
    output_dir: str | Path,
    chunk_size: int = 100,
) -> List[Path]:
    saved_paths: List[Path] = []
    chunk_samples: List[Dict[str, Any]] = []
    chunk_idx = 0

    model.eval()

    with torch.inference_mode():
        for batch in tqdm(loader, desc="Extracting single outputs"):
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            single_inter = model.encode_single_image_intermediates(pixel_values)

            image_emb = single_inter["global_embedding"].cpu()
            image_patch_proj = single_inter["patch_proj"].cpu()
            spatial_attentions = _cpu_clone_tuple_tensors(single_inter["attentions"])

            # flatten [B, H, W, D] -> [B, P, D]
            B, H, W, D = image_patch_proj.shape
            image_patch_proj = image_patch_proj.view(B, H * W, D)

            text_inter = model.encode_text_intermediates(input_ids, attention_mask)
            text_emb = text_inter["cls_projected_embedding"].cpu()

            last_hidden_state = text_inter["last_hidden_state"]
            if last_hidden_state is None:
                raise RuntimeError("No text hidden states returned.")
            text_token_proj = model.project_text_hidden_states(last_hidden_state).cpu()

            text_attentions = _cpu_clone_tuple_tensors(text_inter["attentions"])
            text_hidden_states = _cpu_clone_tuple_tensors(text_inter["hidden_states"])

            sim = torch.sum(image_emb * text_emb, dim=1)
            tokens_batch = _tokenize_batch_to_strings(tokenizer, batch["input_ids"], batch["attention_mask"])

            bs = image_emb.shape[0]
            for i in range(bs):
                per_sample_text_attn = None
                if text_attentions is not None:
                    per_sample_text_attn = [layer[i].clone() for layer in text_attentions]

                per_sample_text_hidden = None
                if text_hidden_states is not None:
                    per_sample_text_hidden = [layer[i].clone() for layer in text_hidden_states]
                
                per_sample_spatial_attn = None              # <-- add this
                if spatial_attentions is not None:           # <-- 
                    per_sample_spatial_attn = [layer[i].clone() for layer in spatial_attentions]


                sample_labels = None
                if "labels" in batch:
                    sample_labels = {
                        k: float(batch["labels"][k][i].item())
                        for k in batch["labels"]
                    }

                chunk_samples.append({
                    "subject_id": batch["meta"]["subject_id"][i],
                    "study_id": batch["meta"]["study_id"][i],
                    "dicom_id": batch["meta"]["dicom_id"][i],
                    "image_path": batch["meta"]["image_path"][i],
                    "report_path": batch["meta"]["report_path"][i],
                    "report_text": batch["meta"]["report_text"][i],
                    "labels": sample_labels,
                    
                    "image_emb": image_emb[i].clone(),
                    "text_emb": text_emb[i].clone(),
                    "image_text_similarity": sim[i].item(),
                    
                    "image_patch_proj": image_patch_proj[i].clone(),
                    "text_token_proj": text_token_proj[i].clone(),
                    
                    "tokens": tokens_batch[i],
                    "input_ids": batch["input_ids"][i].clone(),
                    "attention_mask": batch["attention_mask"][i].clone(),
                    
                    "text_attentions": per_sample_text_attn,
                    "text_hidden_states": per_sample_text_hidden,
                    "spatial_attentions": per_sample_spatial_attn,
                })

                if len(chunk_samples) >= chunk_size:
                    chunk_path = save_single_chunk(chunk_samples, output_dir, chunk_idx)
                    saved_paths.append(chunk_path)
                    chunk_samples = []
                    chunk_idx += 1
                    _cleanup_after_batch(device)

            del pixel_values, input_ids, attention_mask
            del image_emb, image_patch_proj, text_inter, text_emb, text_token_proj
            del text_attentions, text_hidden_states, sim
            _cleanup_after_batch(device)

    if chunk_samples:
        chunk_path = save_single_chunk(chunk_samples, output_dir, chunk_idx)
        saved_paths.append(chunk_path)
        chunk_samples = []
        _cleanup_after_batch(device)

    return saved_paths


def extract_temporal_outputs_chunked(
    model,
    loader,
    tokenizer,
    device: str,
    output_dir: str | Path,
    chunk_size: int = 50,
) -> List[Path]:
    saved_paths: List[Path] = []
    chunk_samples: List[Dict[str, Any]] = []
    chunk_idx = 0

    model.eval()

    with torch.inference_mode():
        for batch in tqdm(loader, desc="Extracting temporal outputs"):
            prior_pixel_values = batch["prior_pixel_values"].to(device)
            current_pixel_values = batch["current_pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            prior_emb = model.encode_image(prior_pixel_values).cpu()
            current_emb = model.encode_image(current_pixel_values).cpu()

            prior_patch_proj = model.encode_image_patches(prior_pixel_values).cpu()
            current_patch_proj = model.encode_image_patches(current_pixel_values).cpu()

            # flatten [B, H, W, D] -> [B, P, D]
            B, H, W, D = prior_patch_proj.shape
            prior_patch_proj = prior_patch_proj.view(B, H * W, D)

            B, H, W, D = current_patch_proj.shape
            current_patch_proj = current_patch_proj.view(B, H * W, D)

            text_inter = model.encode_text_intermediates(input_ids, attention_mask)
            text_emb = text_inter["cls_projected_embedding"].cpu()

            last_hidden_state = text_inter["last_hidden_state"]
            if last_hidden_state is None:
                raise RuntimeError("No text hidden states returned.")
            text_token_proj = model.project_text_hidden_states(last_hidden_state).cpu()

            text_attentions = _cpu_clone_tuple_tensors(text_inter["attentions"])
            text_hidden_states = _cpu_clone_tuple_tensors(text_inter["hidden_states"])

            temporal_inter = model.encode_temporal_intermediates(
                current_images=current_pixel_values,
                previous_images=prior_pixel_values,
            )

            temporal_global_emb = temporal_inter["temporal_global_embedding"]
            if temporal_global_emb is not None:
                temporal_global_emb = temporal_global_emb.cpu()

            current_patch_features = temporal_inter["current_patch_features"].cpu()
            temporal_patch_proj = temporal_inter["temporal_patch_proj"].cpu()

            # flatten [B, H, W, D] -> [B, P, D]
            B, H, W, D = temporal_patch_proj.shape
            temporal_patch_proj = temporal_patch_proj.view(B, H * W, D)

            temporal_attentions_cpu = [
                a.detach().cpu().clone() for a in temporal_inter["temporal_attentions"]
            ]

            prior_text_sim = torch.sum(prior_emb * text_emb, dim=1)
            current_text_sim = torch.sum(current_emb * text_emb, dim=1)
            prior_current_sim = torch.sum(prior_emb * current_emb, dim=1)

            tokens_batch = _tokenize_batch_to_strings(tokenizer, batch["input_ids"], batch["attention_mask"])

            bs = prior_emb.shape[0]
            for i in range(bs):
                per_sample_text_attn = None
                if text_attentions is not None:
                    per_sample_text_attn = [layer[i].clone() for layer in text_attentions]

                per_sample_text_hidden = None
                if text_hidden_states is not None:
                    per_sample_text_hidden = [layer[i].clone() for layer in text_hidden_states]

                per_sample_temporal_attn = [layer[i].clone() for layer in temporal_attentions_cpu]

                sample_labels = None
                if "labels" in batch:
                    sample_labels = {
                        k: float(batch["labels"][k][i].item())
                        for k in batch["labels"]
                    }

                chunk_samples.append({
                    "subject_id": batch["meta"]["subject_id"][i],
                    "prior_study_id": batch["meta"]["prior_study_id"][i],
                    "prior_dicom_id": batch["meta"]["prior_dicom_id"][i],
                    "current_study_id": batch["meta"]["current_study_id"][i],
                    "current_dicom_id": batch["meta"]["current_dicom_id"][i],
                    "gap_days": batch["meta"]["gap_days"][i],
                    "prior_image_path": batch["meta"]["prior_image_path"][i],
                    "current_image_path": batch["meta"]["current_image_path"][i],
                    "report_path": batch["meta"]["report_path"][i],
                    "current_report_text": batch["meta"]["current_report_text"][i],
                    "labels": sample_labels,
                    "prior_emb": prior_emb[i].clone(),
                    "current_emb": current_emb[i].clone(),
                    "text_emb": text_emb[i].clone(),
                    "prior_text_similarity": prior_text_sim[i].item(),
                    "current_text_similarity": current_text_sim[i].item(),
                    "prior_current_similarity": prior_current_sim[i].item(),
                    "prior_patch_proj": prior_patch_proj[i].clone(),
                    "current_patch_proj": current_patch_proj[i].clone(),
                    "current_patch_features": current_patch_features[i].clone(),
                    "temporal_global_emb": (
                        temporal_global_emb[i].clone() if temporal_global_emb is not None else None
                    ),
                    "temporal_patch_proj": temporal_patch_proj[i].clone(),
                    "text_token_proj": text_token_proj[i].clone(),
                    "tokens": tokens_batch[i],
                    "input_ids": batch["input_ids"][i].clone(),
                    "attention_mask": batch["attention_mask"][i].clone(),
                    "temporal_attentions": per_sample_temporal_attn,
                    "text_attentions": per_sample_text_attn,
                    "text_hidden_states": per_sample_text_hidden,
                })

                if len(chunk_samples) >= chunk_size:
                    chunk_path = save_temporal_chunk(chunk_samples, output_dir, chunk_idx)
                    saved_paths.append(chunk_path)
                    chunk_samples = []
                    chunk_idx += 1
                    _cleanup_after_batch(device)

            del prior_pixel_values, current_pixel_values, input_ids, attention_mask
            del prior_emb, current_emb, prior_patch_proj, current_patch_proj
            del text_inter, text_emb, text_token_proj, text_attentions, text_hidden_states
            del temporal_inter, current_patch_features, temporal_patch_proj, temporal_attentions_cpu
            if temporal_global_emb is not None:
                del temporal_global_emb
            del prior_text_sim, current_text_sim, prior_current_sim
            _cleanup_after_batch(device)

    if chunk_samples:
        chunk_path = save_temporal_chunk(chunk_samples, output_dir, chunk_idx)
        saved_paths.append(chunk_path)
        chunk_samples = []
        _cleanup_after_batch(device)

    return saved_paths


def run_biovil_t_single_extraction(
    single_csv_path: str,
    model_name: str = "microsoft/BiomedVLP-BioViL-T",
    output_root: str = "embeddings/biovil_t",
    device: str | None = None,
    single_batch_size: int = 8,
    num_workers: int = 0,
    chunk_size: int = 100,
) -> Dict[str, Any]:
    if device is None:
        device = get_device()

    out_dirs = ensure_output_dirs(output_root)

    single_ds = SingleInferenceDataset(single_csv_path)
    image_processor, tokenizer = build_biovil_preprocessors(model_name=model_name)
    single_loader = build_single_loader(
        single_ds=single_ds,
        image_processor=image_processor,
        tokenizer=tokenizer,
        single_batch_size=single_batch_size,
        num_workers=num_workers,
    )

    model = load_biovil_t(model_name=model_name, device=device)

    chunk_paths = extract_single_outputs_chunked(
        model=model,
        loader=single_loader,
        tokenizer=tokenizer,
        device=device,
        output_dir=out_dirs["single_chunks"],
        chunk_size=chunk_size,
    )

    manifest = {
        "split": "single",
        "model_name": model_name,
        "device": device,
        "count": len(single_ds),
        "chunk_size": chunk_size,
        "chunk_paths": [str(p) for p in chunk_paths],
    }
    manifest_path = save_manifest(manifest, out_dirs["single"] / "single_manifest.pt")

    return {
        "device": device,
        "single_count": len(single_ds),
        "single_manifest_path": manifest_path,
        "single_chunk_paths": chunk_paths,
        "output_root": out_dirs["root"],
    }


def run_biovil_t_temporal_extraction(
    temporal_csv_path: str,
    model_name: str = "microsoft/BiomedVLP-BioViL-T",
    output_root: str = "embeddings/biovil_t",
    device: str | None = None,
    temporal_batch_size: int = 4,
    num_workers: int = 0,
    chunk_size: int = 50,
) -> Dict[str, Any]:
    if device is None:
        device = get_device()

    out_dirs = ensure_output_dirs(output_root)

    temporal_ds = TemporalDataset(temporal_csv_path)
    image_processor, tokenizer = build_biovil_preprocessors(model_name=model_name)
    temporal_loader = build_temporal_loader(
        temporal_ds=temporal_ds,
        image_processor=image_processor,
        tokenizer=tokenizer,
        temporal_batch_size=temporal_batch_size,
        num_workers=num_workers,
    )

    model = load_biovil_t(model_name=model_name, device=device)

    chunk_paths = extract_temporal_outputs_chunked(
        model=model,
        loader=temporal_loader,
        tokenizer=tokenizer,
        device=device,
        output_dir=out_dirs["temporal_chunks"],
        chunk_size=chunk_size,
    )

    manifest = {
        "split": "temporal",
        "model_name": model_name,
        "device": device,
        "count": len(temporal_ds),
        "chunk_size": chunk_size,
        "chunk_paths": [str(p) for p in chunk_paths],
    }
    manifest_path = save_manifest(manifest, out_dirs["temporal"] / "temporal_manifest.pt")

    return {
        "device": device,
        "temporal_count": len(temporal_ds),
        "temporal_manifest_path": manifest_path,
        "temporal_chunk_paths": chunk_paths,
        "output_root": out_dirs["root"],
    }


def run_biovil_t_extraction(
    single_csv_path: str,
    temporal_csv_path: str,
    model_name: str = "microsoft/BiomedVLP-BioViL-T",
    output_root: str = "embeddings/biovil_t",
    device: str | None = None,
    single_batch_size: int = 8,
    temporal_batch_size: int = 4,
    num_workers: int = 0,
    single_chunk_size: int = 100,
    temporal_chunk_size: int = 50,
) -> Dict[str, Any]:
    single_result = run_biovil_t_single_extraction(
        single_csv_path=single_csv_path,
        model_name=model_name,
        output_root=output_root,
        device=device,
        single_batch_size=single_batch_size,
        num_workers=num_workers,
        chunk_size=single_chunk_size,
    )

    temporal_result = run_biovil_t_temporal_extraction(
        temporal_csv_path=temporal_csv_path,
        model_name=model_name,
        output_root=output_root,
        device=device,
        temporal_batch_size=temporal_batch_size,
        num_workers=num_workers,
        chunk_size=temporal_chunk_size,
    )

    return {
        "device": single_result["device"],
        "single_count": single_result["single_count"],
        "temporal_count": temporal_result["temporal_count"],
        "single_manifest_path": single_result["single_manifest_path"],
        "temporal_manifest_path": temporal_result["temporal_manifest_path"],
        "single_chunk_paths": single_result["single_chunk_paths"],
        "temporal_chunk_paths": temporal_result["temporal_chunk_paths"],
        "output_root": single_result["output_root"],
    }


def load_or_extract_biovil_t_single_embeddings(
    single_csv_path: str,
    model_name: str = "microsoft/BiomedVLP-BioViL-T",
    output_root: str = "embeddings/biovil_t",
    device: str | None = None,
    single_batch_size: int = 8,
    num_workers: int = 0,
    chunk_size: int = 100,
    force_recompute: bool = False,
):
    out_dirs = ensure_output_dirs(output_root)
    manifest_path = out_dirs["single"] / "single_manifest.pt"

    need_extract = force_recompute or not manifest_path.exists()

    if need_extract:
        print("Single BioViL-T embeddings not found. Running single extraction...")
        run_biovil_t_single_extraction(
            single_csv_path=single_csv_path,
            model_name=model_name,
            output_root=output_root,
            device=device,
            single_batch_size=single_batch_size,
            num_workers=num_workers,
            chunk_size=chunk_size,
        )
    else:
        print("Single BioViL-T embeddings found. Loading manifest from disk...")

    manifest = torch.load(manifest_path)

    return {
        "device": device if device is not None else get_device(),
        "single_manifest_path": manifest_path,
        "single_chunk_paths": [Path(p) for p in manifest["chunk_paths"]],
        "single_manifest": manifest,
    }


def load_or_extract_biovil_t_temporal_embeddings(
    temporal_csv_path: str,
    model_name: str = "microsoft/BiomedVLP-BioViL-T",
    output_root: str = "embeddings/biovil_t",
    device: str | None = None,
    temporal_batch_size: int = 4,
    num_workers: int = 0,
    chunk_size: int = 50,
    force_recompute: bool = False,
):
    out_dirs = ensure_output_dirs(output_root)
    manifest_path = out_dirs["temporal"] / "temporal_manifest.pt"

    need_extract = force_recompute or not manifest_path.exists()

    if need_extract:
        print("Temporal BioViL-T embeddings not found. Running temporal extraction...")
        run_biovil_t_temporal_extraction(
            temporal_csv_path=temporal_csv_path,
            model_name=model_name,
            output_root=output_root,
            device=device,
            temporal_batch_size=temporal_batch_size,
            num_workers=num_workers,
            chunk_size=chunk_size,
        )
    else:
        print("Temporal BioViL-T embeddings found. Loading manifest from disk...")

    manifest = torch.load(manifest_path)

    return {
        "device": device if device is not None else get_device(),
        "temporal_manifest_path": manifest_path,
        "temporal_chunk_paths": [Path(p) for p in manifest["chunk_paths"]],
        "temporal_manifest": manifest,
    }



def load_single_samples(single_chunk_paths: Sequence[str | Path]) -> List[Dict[str, Any]]:
    samples = []
    for chunk_path in single_chunk_paths:
        chunk = torch.load(chunk_path, map_location="cpu")

        n = len(chunk["metadata"])
        for i in range(n):
            samples.append({
                "metadata": chunk["metadata"][i],
                "labels": chunk["labels"][i],
                "image_emb": chunk["image_emb"][i].float(),
                "text_emb": chunk["text_emb"][i].float(),
                "image_patch_proj": chunk["image_patch_proj"][i].float(),
                "text_token_proj": chunk["text_token_proj"][i].float() if chunk["text_token_proj"][i] is not None else None,
                "tokens": chunk["tokens"][i],
            })
    return samples


def select_single_samples(
    samples: Sequence[Dict[str, Any]],
    concept_name: str,
    positive_label: float = 1.0,
    max_positive: int | None = None,
    max_negative: int | None = None,
    seed: int = 42,
):
    rng = random.Random(seed)

    positives = []
    negatives = []

    for s in samples:
        labels = s.get("labels")
        if labels is None or concept_name not in labels:
            continue

        if float(labels[concept_name]) == positive_label:
            positives.append(s)
        else:
            negatives.append(s)

    if max_positive is not None and len(positives) > max_positive:
        positives = rng.sample(positives, max_positive)

    if max_negative is not None and len(negatives) > max_negative:
        negatives = rng.sample(negatives, max_negative)

    return positives, negatives



def load_temporal_samples(temporal_chunk_paths):
    samples = []

    for chunk_path in temporal_chunk_paths:
        chunk = torch.load(chunk_path, map_location="cpu")

        n = len(chunk["metadata"])
        for i in range(n):
            sample = {
                "chunk_path": str(chunk_path),
                "sample_idx": i,
                "metadata": chunk["metadata"][i],
                "labels": chunk["labels"][i] if "labels" in chunk else None,
                "prior_emb": chunk["prior_emb"][i].float(),
                "current_emb": chunk["current_emb"][i].float(),
                "text_emb": chunk["text_emb"][i].float(),
                "prior_patch_proj": chunk["prior_patch_proj"][i].float(),
                "current_patch_proj": chunk["current_patch_proj"][i].float(),
                "temporal_patch_proj": chunk["temporal_patch_proj"][i].float(),
            }

            if chunk.get("temporal_global_emb") is not None:
                sample["temporal_global_emb"] = chunk["temporal_global_emb"][i].float()
            else:
                sample["temporal_global_emb"] = None

            if "text_token_proj" in chunk and chunk["text_token_proj"][i] is not None:
                sample["text_token_proj"] = chunk["text_token_proj"][i].float()
            else:
                sample["text_token_proj"] = None

            sample["tokens"] = chunk["tokens"][i]
            sample["input_ids"] = chunk["input_ids"][i]
            sample["attention_mask"] = chunk["attention_mask"][i]
            sample["temporal_attentions"] = chunk["temporal_attentions"][i]

            samples.append(sample)

    return samples


def select_temporal_samples(
    samples,
    concept_name,
    positive_label=1.0,
    max_positive=None,
    max_negative=None,
    seed=42,
):
    rng = random.Random(seed)

    positives = []
    negatives = []

    for s in samples:
        labels = s.get("labels")
        if labels is None or concept_name not in labels:
            continue

        if float(labels[concept_name]) == positive_label:
            positives.append(s)
        else:
            negatives.append(s)

    if max_positive is not None and len(positives) > max_positive:
        positives = rng.sample(positives, max_positive)

    if max_negative is not None and len(negatives) > max_negative:
        negatives = rng.sample(negatives, max_negative)

    return positives, negatives
