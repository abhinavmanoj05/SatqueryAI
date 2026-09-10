import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_reben = _ROOT / "reben-training-scripts"
if _reben.exists() and str(_reben) not in sys.path:
    sys.path.insert(0, str(_reben))

import torch
from reben_publication.BigEarthNetv2_0_ImageClassifier import BigEarthNetv2_0_ImageClassifier

MODEL_NAME = "BIFOLD-BigEarthNetv2-0/resnet18-all-v0.2.0"

model = BigEarthNetv2_0_ImageClassifier.from_pretrained(MODEL_NAME)
model.eval()

channels = model.config.channels
image_size = model.config.image_size
print(f"Model loaded. Expected input channels: {channels}, image size: {image_size}x{image_size}")

x = torch.randn(1, channels, image_size, image_size)
with torch.no_grad():
    y = model(x)
print("Input: ", x.shape)
print("Output: ", y.shape)
print("Setup verified successfully!")
