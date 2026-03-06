from typing import List, Dict, Any
import torch


def collate_biovil_temporal(
    batch: List[Dict[str, Any]],
    image_processor,
    tokenizer,
    max_length: int = 256
):
    prior_images = torch.stack(
        [image_processor(b["prior_image"]) for b in batch], dim=0
    )
    current_images = torch.stack(
        [image_processor(b["current_image"]) for b in batch], dim=0
    )

    texts = [b["text"] for b in batch]
    text_inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    pixel_values = torch.stack([prior_images, current_images], dim=1)  # [B, 2, C, H, W]

    return {
        "pixel_values": pixel_values,
        "input_ids": text_inputs["input_ids"],
        "attention_mask": text_inputs["attention_mask"],
        "meta": {
            "prior_image_path": [b["prior_image_path"] for b in batch],
            "current_image_path": [b["current_image_path"] for b in batch],
            "subject_id": [b.get("subject_id") for b in batch],
            "gap_days": [b.get("gap_days") for b in batch],
        }
    }

def collate_biovil_single(
    batch: List[Dict[str, Any]],
    image_processor,
    tokenizer,
    max_length: int = 256
):
    images = torch.stack(
        [image_processor(b["image"]) for b in batch], dim=0
    )

    texts = [b["text"] for b in batch]
    text_inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    return {
        "pixel_values": images,
        "input_ids": text_inputs["input_ids"],
        "attention_mask": text_inputs["attention_mask"],
        "meta": {
            "image_path": [b["image_path"] for b in batch],
            "subject_id": [b.get("subject_id") for b in batch],
            "study_id": [b.get("study_id") for b in batch],
            "dicom_id": [b.get("dicom_id") for b in batch],
        }
    }