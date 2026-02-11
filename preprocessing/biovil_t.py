from transformers import AutoProcessor, AutoTokenizer

def load_biovil_preprocessor(model_name: str = "microsoft/BiomedVLP-BioViL-T", mode: str = "multimodal"):
    if mode == "multimodal":
        preprocessor = AutoProcessor.from_pretrained(
            model_name=model_name,
            trust_remote_code=True    
        )
    elif mode == "text":
        preprocessor = AutoTokenizer.from_pretrained(
            model_name=model_name,
            trust_remote_code=True
        )
    else:
        raise ValueError("mode should either be 'multimodal' or 'Text'.")
    
    return preprocessor