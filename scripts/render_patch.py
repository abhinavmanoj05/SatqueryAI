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

def stretch(band: np.ndarray) -> np.ndarray:
    p2 = np.percentile(band, 2)
    p98 = np.percentile(band, 98)
    if p98 <= p2:
        return np.zeros_like(band, dtype=np.uint8)
    clipped = np.clip(band, p2, p98)
    return ((clipped - p2) / (p98 - p2) * 255).astype(np.uint8)


def main():
    # True color: B04 (red), B03 (green), B02 (blue) - all 10m, already 120x120
    r = load_band("B04", "tiff")
    g = load_band("B03", "tiff")
    b = load_band("B02", "tiff")

    rgb = np.stack([stretch(r), stretch(g), stretch(b)], axis=-1)
    im = Image.fromarray(rgb, "RGB")
    im = im.resize((480, 480), Image.NEAREST)  # upscale for visibility
    im.save(OUT)
    print("Saved RGB patch to:", OUT)

    # Also render a post-event patch simulating land-use modification for bi-temporal testing
    post_rgb = rgb.copy()
    post_rgb[15:55, 60:110, 0] = 200  # soil/clearing alteration
    post_rgb[15:55, 60:110, 1] = 180
    post_rgb[15:55, 60:110, 2] = 140
    post_out = Path(__file__).resolve().parent / "sample_patch_post.png"
    im_post = Image.fromarray(post_rgb, "RGB").resize((480, 480), Image.NEAREST)
    im_post.save(post_out)
    print("Saved post-change RGB patch to:", post_out)


if __name__ == "__main__":
    main()

