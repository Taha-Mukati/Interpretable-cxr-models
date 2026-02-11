import torch
from pprint import pprint

ckpt = torch.load(
    "/Users/taha/Desktop/UNI/ECM 3401/interpretable-cxr-models/models/cxr_clip/r50_m.tar",
    map_location="cpu",
    weights_only=False,  # important for old checkpoints
)

tokenizer = ckpt["model"]

print("\nmodel top-level keys:")
pprint(tokenizer.keys())



print("Top-level keys:")
pprint(ckpt.keys())
