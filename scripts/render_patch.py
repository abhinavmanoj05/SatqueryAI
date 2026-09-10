"""
Render the sample BigEarthNet v2.0 patch as a true-color RGB image (B04=red, B03=green, B02=blue)
and save it to a PNG so it can be visually inspected.
"""
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = _ROOT / "reben-training-scripts" / "scripts" / "data"
OUT = Path(__file__).resolve().parent / "sample_patch_rgb.png"


def load_band(band: str, ext: str):
    folder = DATA_ROOT / "S2"
    files = [f for f in folder.rglob(f"*_{band}.{ext}")]
    assert len(files) == 1, files
    with rasterio.open(files[0]) as src:
        return src.read(1)


def main():
    # True color: B04 (red), B03 (green), B02 (blue) - all 10m, already 120x120
    r = load_band("B04", "tiff")
    g = load_band("B03", "tiff")
    b = load_band("B02", "tiff")

    # clip to display range and normalize to 8-bit
    def to8bit(x):
        x = np.clip(x, 0, 8000)  # Sentinel-2 reflectance range
        return (x / 8000 * 255).astype("uint8")

    img = np.stack([to8bit(r), to8bit(g), to8bit(b)], axis=-1)
    im = Image.fromarray(img, "RGB")
    im = im.resize((480, 480), Image.NEAREST)  # upscale for visibility
    im.save(OUT)
    print("Saved RGB patch to:", OUT)


if __name__ == "__main__":
    main()
