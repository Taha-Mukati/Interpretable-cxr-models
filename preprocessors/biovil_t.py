from transformers import AutoTokenizer
from torchvision import transforms


def load_biovil_preprocessor(
    model_name: str = "microsoft/BiomedVLP-BioViL-T",
    mode: str = "multimodal"
):
    if mode == "text":
        return AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

    elif mode == "image":
        # Manual image preprocessing
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    elif mode == "multimodal":
        return {
            "image_processor": load_biovil_preprocessor(model_name, mode="image"),
            "tokenizer": load_biovil_preprocessor(model_name, mode="text"),
        }

    else:
        raise ValueError("mode should be 'image', 'text', or 'multimodal'.")