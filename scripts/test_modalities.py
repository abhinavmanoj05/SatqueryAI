"""
scripts/test_modalities.py
==========================
Tests bi-temporal change detection and Optical+SAR fusion using real dataset imagery.
Creates prepared GeoTIFF and PNG test samples in test_samples/ directory so the user
can also use them directly in the UI.
"""

import os
import sys
from pathlib import Path
from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from run_orchestrator import run_satquery


def prepare_sample_datasets():
    samples_dir = _ROOT / "test_samples"
    bitemp_dir = samples_dir / "bitemporal"
    opt_sar_dir = samples_dir / "optical_sar"

    bitemp_dir.mkdir(parents=True, exist_ok=True)
    opt_sar_dir.mkdir(parents=True, exist_ok=True)

    # 1. Prepare Bi-Temporal pair (T0 pre-event and T1 post-event with clearing)
    src_t0 = _ROOT / "scripts" / "sample_patch_rgb.png"
    src_t1 = _ROOT / "scripts" / "sample_patch_post.png"

    img_t0 = Image.open(src_t0).convert("RGB")
    img_t1 = Image.open(src_t1).convert("RGB")

    t0_png = bitemp_dir / "t0_pre_event.png"
    t1_png = bitemp_dir / "t1_post_event.png"
    t0_tif = bitemp_dir / "t0_pre_event.tif"
    t1_tif = bitemp_dir / "t1_post_event.tif"

    img_t0.save(t0_png)
    img_t1.save(t1_png)
    img_t0.save(t0_tif, format="TIFF")
    img_t1.save(t1_tif, format="TIFF")

    # 2. Prepare Optical + SAR pair from BigEarthNet archive
    v2_base = Path("C:/Users/DELLG15/Downloads/archive (2)/v_2")
    src_s2 = v2_base / "agri" / "s2" / "ROIs1868_summer_s2_59_p10.png"
    src_s1 = v2_base / "agri" / "s1" / "ROIs1868_summer_s1_59_p10.png"

    if src_s2.exists() and src_s1.exists():
        img_s2 = Image.open(src_s2).convert("RGB")
        img_s1 = Image.open(src_s1).convert("RGB")

        s2_png = opt_sar_dir / "sentinel2_optical.png"
        s1_png = opt_sar_dir / "sentinel1_sar.png"
        s2_tif = opt_sar_dir / "sentinel2_optical.tif"
        s1_tif = opt_sar_dir / "sentinel1_sar.tif"

        img_s2.save(s2_png)
        img_s1.save(s1_png)
        img_s2.save(s2_tif, format="TIFF")
        img_s1.save(s1_tif, format="TIFF")
    else:
        s2_png, s1_png, s2_tif, s1_tif = None, None, None, None

    return {
        "bitemp_png": (str(t0_png), str(t1_png)),
        "bitemp_tif": (str(t0_tif), str(t1_tif)),
        "opt_sar_png": (str(s2_png), str(s1_png)) if s2_png else None,
        "opt_sar_tif": (str(s2_tif), str(s1_tif)) if s2_tif else None,
    }


def test_bitemporal_pipeline(t0_path: str, t1_path: str, is_tif: bool = False):
    print("\n" + "=" * 65)
    print(f"TESTING BI-TEMPORAL CHANGE DETECTION ({'TIFF' if is_tif else 'PNG'})")
    print(f"T0 (Pre-Event):  {t0_path}")
    print(f"T1 (Post-Event): {t1_path}")
    print("=" * 65)

    files = [
        {"path": t0_path, "modality": "optical", "format": "geotiff" if is_tif else "png"},
        {"path": t1_path, "modality": "optical", "format": "geotiff" if is_tif else "png"},
    ]

    result = run_satquery(
        user_query="What changed between these two dates? Quantify vegetation loss and built-up expansion.",
        uploaded_files=files,
        preferred_model="auto",
    )

    print("\n--- FINAL ANSWER ---")
    print(result.get("final_answer"))

    ev = result.get("visual_evidence", {})
    stats = ev.get("stats", {})
    print("\n--- CHANGE METRICS ---")
    print(f"Percentage Changed:           {stats.get('percentage_changed')}%")
    print(f"Vegetation Loss:              {stats.get('vegetation_loss_percentage')}%")
    print(f"Built-Up Expansion:           {stats.get('built_up_expansion_percentage')}%")
    print(f"Active Hotspots:              {stats.get('active_hotspots')}")
    print(f"Semantic Transition Prior:    {ev.get('transition')}")
    print(f"Heatmap Mask Length:          {len(ev.get('change_mask', ''))} chars (Base64 PNG)")

    trace = result.get("execution_trace", {})
    print("\n--- EXECUTION TRACE ---")
    print(f"Routed Task:   {trace.get('task')}")
    print(f"Models Used:   {trace.get('models_used')}")
    print(f"Duration:      {trace.get('duration_ms')} ms")
    print(f"Confidence:    {result.get('confidence')}")

    assert trace.get("task") == "change_detection"
    assert stats.get("percentage_changed", 0) > 0
    print("\n>>> BI-TEMPORAL CHANGE DETECTION TEST: SUCCESSFUL")


def test_optical_sar_fusion_pipeline(opt_path: str, sar_path: str, is_tif: bool = False):
    print("\n" + "=" * 65)
    print(f"TESTING OPTICAL + SAR FUSION CLASSIFICATION ({'TIFF' if is_tif else 'PNG'})")
    print(f"Optical (S2): {opt_path}")
    print(f"SAR (S1):     {sar_path}")
    print("=" * 65)

    files = [
        {"path": opt_path, "modality": "optical", "format": "geotiff" if is_tif else "png"},
        {"path": sar_path, "modality": "sar", "format": "geotiff" if is_tif else "png"},
    ]

    result = run_satquery(
        user_query="Jointly analyze this Optical and SAR imagery to identify land cover and moisture penetration.",
        uploaded_files=files,
        preferred_model="auto",
    )

    print("\n--- FINAL ANSWER ---")
    print(result.get("final_answer"))

    ev = result.get("visual_evidence", {})
    top_k = ev.get("top_k", [])
    print("\n--- MULTISPECTRAL TOP-K LAND COVER (ViT-Base 12-Band) ---")
    for item in top_k:
        print(f"• {item['class']}: {item['probability']*100:.2f}%")

    trace = result.get("execution_trace", {})
    print("\n--- EXECUTION TRACE ---")
    print(f"Routed Task:   {trace.get('task')}")
    print(f"Models Used:   {trace.get('models_used')}")
    print(f"Duration:      {trace.get('duration_ms')} ms")
    print(f"Confidence:    {result.get('confidence')}")

    assert trace.get("task") == "fusion"
    assert len(top_k) > 0
    print("\n>>> OPTICAL + SAR FUSION TEST: SUCCESSFUL")


if __name__ == "__main__":
    paths = prepare_sample_datasets()
    print("Prepared Sample Datasets in:", _ROOT / "test_samples")

    # Run Bi-Temporal test with TIFF
    t0_tif, t1_tif = paths["bitemp_tif"]
    test_bitemporal_pipeline(t0_tif, t1_tif, is_tif=True)

    # Run Optical + SAR test with TIFF
    if paths["opt_sar_tif"]:
        opt_tif, sar_tif = paths["opt_sar_tif"]
        test_optical_sar_fusion_pipeline(opt_tif, sar_tif, is_tif=True)
