from transformers import AutoTokenizer
from health_multimodal.image.data.transforms import create_chest_xray_transform_for_inference


def load_biovil_preprocessor(
    model_name: str = "microsoft/BiomedVLP-BioViL-T",
    mode: str = "multimodal",
):
    if mode == "text":
        return AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

    elif mode == "image":
        return create_chest_xray_transform_for_inference(
            resize=512,
            center_crop_size=448,
        )

    elif mode == "multimodal":
        return {
            "image_processor": load_biovil_preprocessor(model_name=model_name, mode="image"),
            "tokenizer": load_biovil_preprocessor(model_name=model_name, mode="text"),
        }

    else:
        raise ValueError("mode should be 'image', 'text', or 'multimodal'.")