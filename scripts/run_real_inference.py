"""
Real-data inference for BIFOLD-BigEarthNetv2-0/resnet18-all-v0.2.0
using the sample BigEarthNet v2.0 patch bundled in reben-training-scripts/scripts/data.

The model expects 12 channels in this order:
    [VV, VH, B02, B03, B04, B08, B05, B06, B07, B8A, B11, B12]
(2 Sentinel-1 radar bands first, then the 10m S2 bands, then the 20m S2 bands),
each upsampled/normalized to 120x120.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_reben = _ROOT / "reben-training-scripts"
if _reben.exists() and str(_reben) not in sys.path:
    sys.path.insert(0, str(_reben))

import numpy as np
import rasterio
import torch
from configilm.extra.BENv2_utils import STANDARD_BANDS, stack_and_interpolate, band_combi_to_mean_std, NEW_LABELS

from reben_publication.BigEarthNetv2_0_ImageClassifier import BigEarthNetv2_0_ImageClassifier

MODEL_NAME = "BIFOLD-BigEarthNetv2-0/resnet18-all-v0.2.0"
DATA_ROOT = _ROOT / "reben-training-scripts" / "scripts" / "data"


def load_bands(band: str):
    if band in STANDARD_BANDS["S1"]:
        folder = DATA_ROOT / "S1"
        files = [f for f in folder.rglob("*.tif")]
        ext = ".tif"
    else:
        folder = DATA_ROOT / "S2"
        files = [f for f in folder.rglob("*.tif*")]
        ext = ".tiff"

    matching = [f for f in files if f.name.endswith(f"_{band}{ext}")]
    assert len(matching) == 1, f"Expected 1 file for band {band}, found {len(matching)}: {[f.name for f in matching]}"
    with rasterio.open(matching[0]) as src:
        return src.read(1)  # 2D array for one band


def main():
    torch.manual_seed(42)

    # 1. Load the model
    print(f"Loading model {MODEL_NAME} ...")
    model = BigEarthNetv2_0_ImageClassifier.from_pretrained(MODEL_NAME)
    model.eval()

    img_size = model.config.image_size
    print(f"Model expects {model.config.channels} channels @ {img_size}x{img_size}")

    # 2. Load each requested band as a 2D array
    bands = list(STANDARD_BANDS[12])  # [VV, VH, B02, B03, B04, B08, B05, B06, B07, B8A, B11, B12]
    print("Loading bands:", bands)
    data = {b: load_bands(b) for b in bands}

    # 3. Stack and interpolate to 120x120 in the exact model band order
    img = stack_and_interpolate(data, order=bands, img_size=img_size, upsample_mode="nearest")
    print("Stacked raw tensor shape:", tuple(img.shape))

    # 4. Normalize using the same stats configilm uses for BENv2 training
    mean, std = band_combi_to_mean_std(bands, interpolation="120_nearest")
    mean = torch.as_tensor(mean, dtype=torch.float32).view(-1, 1, 1)
    std = torch.as_tensor(std, dtype=torch.float32).view(-1, 1, 1)
    img = (img - mean) / std
    img = img.unsqueeze(0).float()  # add batch dim -> (1, C, 120, 120)

    # 5. Run inference
    with torch.no_grad():
        logits = model(img)
    probs = torch.sigmoid(logits).squeeze(0)  # 19 values

    # 6. Print results
    print("\n================ Predictions ================")
    sorted_idx = probs.argsort(descending=True)
    for rank, i in enumerate(sorted_idx):
        print(f"{rank + 1:>2}. {NEW_LABELS[i]:<60} {probs[i].item():.4f}")
    print("=============================================")


if __name__ == "__main__":
    main()
