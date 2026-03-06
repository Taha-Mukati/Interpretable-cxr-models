import torch
from pprint import pprint

ckpt = torch.load(
    "/Users/taha/Desktop/UNI/ECM 3401/interpretable-cxr-models/models/cxr_clip/r50_m.tar",
    map_location="cpu",
    weights_only=False,  # important for old checkpoints
)

cfg = ckpt["config"]

print(type(cfg))
print(cfg)