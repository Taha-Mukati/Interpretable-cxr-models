from transformers import AutoModel, AutoTokenizer, AutoProcessor

def load_biovil_t(model_name: str = "microsoft/BiomedVLP-BioViL-T",
                  mode: str = "multimodal",
                  device: str = "cpu"
                  ):
    
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True
    ).to(device)
    model.eval()

    
    return model