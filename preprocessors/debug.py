import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CXR_CLIP_REPO = PROJECT_ROOT / "models" / "cxr_clip" / "cxr-clip"

print("PROJECT_ROOT:", PROJECT_ROOT)
print("CXR_CLIP_REPO:", CXR_CLIP_REPO)
print("Exists:", CXR_CLIP_REPO.exists())

if str(CXR_CLIP_REPO) not in sys.path:
    sys.path.insert(0, str(CXR_CLIP_REPO))

print("sys.path[0]:", sys.path[0])

import cxrclip.data.data_utils as du

print("Loaded:", du.__file__)
print("Has load_transform:", hasattr(du, "load_transform"))
print("Has transform_image:", hasattr(du, "transform_image"))