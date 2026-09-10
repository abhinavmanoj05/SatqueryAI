# SatQuery AI

**SIH Problem Statement:** SIH26167  
**Title:** SatQuery AI — An Interactive Vision-Language Assistant for Multimodal Remote Sensing Image Analysis through Text Queries  
**Organisation:** Indian Space Research Organisation (ISRO)  
**Theme:** Space Technology | **Category:** Software  

---

## 📌 Overview

**SatQuery AI** is an agentic vision-language assistant designed to analyze single and paired remote-sensing satellite imagery (Sentinel-1 SAR, Sentinel-2 Optical, and ISRO Cartosat/RISAT data) through natural language queries.

Rather than relying on a single generic VLM, SatQuery AI leverages a **query-driven agentic framework** orchestrating specialist models:
1. **ResNet-18** (`BIFOLD-BigEarthNetv2-0/resnet18-all-v0.2.0`): 12-channel native Optical-SAR classifier generating land-cover sensor priors.
2. **PaliGemma-3B**: Remote sensing scene captioning and description.
3. **InternVL2-8B**: High-resolution Visual Question Answering (VQA) and text-guided region grounding.
4. **Qwen2-VL-7B**: Bi-temporal multi-temporal change understanding and change VQA.
5. **CDVQA Baseline**: Pixel-level change mask detection and spatial change statistics.

---

## 🗂️ Project Organization

```
SatqueryAI/
├── context.md                    # Project specification & single source of truth
├── README.md                     # Project documentation
├── requirements.txt              # Core ML, geospatial, agentic, and UI dependencies
├── launch.bat                    # Interactive Windows launcher
├── ui.py                         # Gradio Web UI for model exploration & interactive testing
│
├── tools/                        # Specialist model tools & inference wrappers
│   ├── __init__.py               # Package exports
│   ├── resnet_tool.py            # ResNet-18 Optical-SAR 12-channel prior generator
│   ├── paligemma_tool.py         # PaliGemma-3B scene captioner
│   ├── internvl2_tool.py         # InternVL2-8B VQA and bounding box visual grounding
│   ├── qwen2vl_tool.py           # Qwen2-VL-7B bi-temporal change VQA
│   └── cdvqa_tool.py             # Dual-temporal change baseline and mask generator
│
├── benchmarks/                   # Benchmark suites and evaluation results
│   ├── test_all_models.py        # Master test harness across all 5 specialist models
│   └── benchmark_results.json    # Auditable evaluation output
│
├── tests/                        # Verification and sanity tests
│   └── test_resnet18_all.py      # ResNet-18 12-channel tensor verification
│
├── scripts/                      # Utility scripts & sample data visualization
│   ├── render_patch.py           # Sentinel-2 true-color RGB patch renderer
│   ├── run_real_inference.py     # End-to-end ResNet-18 12-channel patch inference
│   └── sample_patch_rgb.png      # Rendered sample RGB patch
│
├── docs/                         # Specifications and documentation
│   └── Requirement.pdf           # SIH / ISRO problem statement requirements
│
├── fusion/                       # Cross-modal fusion & tiling modules (in development)
├── output/                       # Visual evidence renderers & audit logs (in development)
├── fine_tuning/                  # QLoRA fine-tuning pipelines (PaliGemma, Qwen2-VL, InternVL2)
├── models/                       # Checkpoints & LoRA adapter weights
└── reben-training-scripts/       # BIFOLD BigEarthNet v2.0 official toolkit & sample data
```

---

## 🚀 Quick Start

### 1. Windows Interactive Launcher
Double click `launch.bat` or run:
```cmd
launch.bat
```
Options provided:
- `[1]` Launch Gradio Web UI (`ui.py`)
- `[2]` Run benchmark tests (`benchmarks/test_all_models.py`)
- `[3]` Quick ResNet-18 inference (`scripts/run_real_inference.py`)

### 2. Manual Commands
```bash
# Activate environment
ben_venv\Scripts\activate

# Launch Web UI
python ui.py

# Run benchmark suite across all tools
python benchmarks/test_all_models.py

# Run ResNet-18 sample patch inference
python scripts/run_real_inference.py
```

---

## 🧪 Benchmark Verification

To evaluate all specialist tools and generate `benchmarks/benchmark_results.json`:
```bash
python benchmarks/test_all_models.py
```
All tools produce observable execution traces with latency, confidence metrics, and device logs.
