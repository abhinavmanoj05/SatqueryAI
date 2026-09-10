# SatQuery AI — Project Context

**SIH Problem Statement:** SIH26167  
**Title:** SatQuery AI — An Interactive Vision-Language Assistant for Multimodal Remote Sensing Image Analysis through Text Queries  
**Organisation:** Indian Space Research Organisation (ISRO)  
**Category:** Software | **Theme:** Space Technology  
**Deadline:** 30 September 2026  

---

## 1. What We Are Building

An **agentic vision-language assistant** that analyses single and paired remote-sensing images through natural-language queries. Unlike generic VLMs, SatQuery AI uses a **query-driven agentic framework** that selects, sequences, and executes remote-sensing specialist models, validates inputs, combines outputs, and returns evidence-grounded responses with visual explanations.

**Core novelty:** Instead of one monolithic VLM, a LangGraph-based agentic controller routes each query to the right specialist tool(s), fuses their outputs, and produces an auditable execution trace.

---

## 2. Mandatory Functional Scope (Must Demonstrate All)

| # | Requirement | Our Implementation |
|---|-------------|-------------------|
| 1 | **Single-image VQA** | InternVL2-8B (fine-tuned on VRSBench + RSVQA + BigEarthNet) |
| 2 | **Second single-image task** | PaliGemma-3B captioning **or** InternVL2 grounding |
| 3 | **Multitemporal change understanding** | Qwen2-VL-7B (fine-tuned on CDVQA) + CDVQA baseline |
| 4 | **Optical–SAR paired analysis** | ResNet-18 (frozen) generates sensor prior → injected into VLM prompt |
| 5 | **Agentic orchestration** | LangGraph `StateGraph` with query parser, validator, router, specialist nodes, output combinator |

**Critical rule:** Only the **observable execution trace** is evaluated. Internal reasoning is neither required nor evaluated. Every response must include: selected task, models/tools used, permitted parameters, and outputs.

---

## 3. Defined Input Scope

| Input Type | Description | Format |
|------------|-------------|--------|
| **Single image** | One optical/multispectral or SAR image | GeoTIFF, TIFF |
| **Cross-modal pair** | Co-registered optical + SAR, same area | GeoTIFF, TIFF |
| **Bi-temporal pair** | Two spatially corresponding images, different dates | GeoTIFF, TIFF |
| **Benchmark inputs** | PNG/JPEG accepted **only** for VRSBench, RSVQA, CDVQA | PNG, JPEG |

---

## 4. Datasets

### 4.1 Training
| Dataset | Purpose | Notes |
|---------|---------|-------|
| **BigEarthNet v2.0** | Multi-sensor adaptation | 549,488 S1+S2 pairs, 12-channel input |
| **BigEarthNet.txt** | Image–text fine-tuning | 464,044 images, 9.6M text annotations |
| **RSVQA / RSVQAxBEN** | VQA training | ~15M samples |
| **CDVQA** | Change VQA training | 2,968 bi-temporal pairs, 122k QA pairs |

### 4.2 Evaluation (Prescribed)
| Benchmark | Task | Metric |
|-----------|------|--------|
| **VRSBench** | Captioning, Grounding, VQA | BLEU, IoU, Accuracy |
| **RSVQA** | VQA | Accuracy |
| **CDVQA** | Change VQA | Accuracy |
| **ISRO/SAC set** | Optical–SAR joint analysis | Cartosat-2S + RISAT pairs (annotations hidden) |

---

## 5. Model Registry

| Tool | Task | Fine-Tuning | Training Mix | Status |
|------|------|-------------|--------------|--------|
| **ResNet-18** (`BIFOLD-BigEarthNetv2-0/resnet18-all-v0.2.0`) | Optical–SAR land-cover classification | ❄️ Frozen | BigEarthNet v2.0 (pre-trained) | ✅ Tested |
| **PaliGemma-3B** | Scene captioning | 🔥 QLoRA (attn layers) | BigEarthNet.txt caption splits (100%) | 🔄 In progress |
| **InternVL2-8B** | VQA + grounding | 🔥 QLoRA | VRSBench (40%) + RSVQA (30%) + BEN cls (30%) | ⏳ Pending |
| **Qwen2-VL-7B** | Bi-temporal change VQA | 🔥 QLoRA (4-bit) | CDVQA (100%) | ⏳ Pending |
| **CDVQA Baseline** | Pixel-level change detection | ❄️ Frozen | — | ⏳ Pending |

### 5.1 ResNet-18 I/O (Working)
- **Input:** `(B, 12, 120, 120)` — channel order: `[VV, VH, B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12]`
- **Output:** `(B, 19)` logits → sigmoid for multi-label probabilities
- **Classes:** 19 CORINE land-cover classes (alphabetical order)
- **20m bands upsampled to 120×120** (bilinear/cubic)
- **Expected performance:** Macro AP ≈ 0.71, Macro F1 ≈ 0.66

### 5.2 VLM I/O
- All VLMs accept **3-channel RGB only**
- BigEarthNet: extract bands 4,3,2 (R,G,B)
- SAR: create false-color RGB (Red=VH, Green=VV, Blue=VH/VV ratio)

---

## 6. System Architecture — Five Layers

```
┌─────────────────────────────────────────────────────────────────┐
│ LAYER 1: PRESENTATION & INPUT GATEWAY (Gradio/Streamlit)       │
│  • File uploader  • Format validator  • Sensor classifier       │
└─────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ LAYER 2: AGENTIC ORCHESTRATOR (LangGraph StateGraph)           │
│  • Query Parser → Intent JSON                                  │
│  • Validation Gate → input compatibility                       │
│  • Task Router → conditional edges                             │
│  • Audit Logger → execution trace                              │
└─────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ LAYER 3: TOOL REGISTRY (Specialist Models)                     │
│  ResNet-18 | PaliGemma | InternVL2 | Qwen2-VL | CDVQA          │
└─────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ LAYER 4: CROSS-MODAL FUSION BRIDGE                             │
│  • Prompt Injector (ResNet prior → VLM prompt)                 │
│  • Tiling Manager (1024×1024 for high-res ISRO images)         │
└─────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ LAYER 5: EVIDENCE GROUNDING & OUTPUT COMBINATOR                │
│  • Spatial Renderer  • Confidence Estimator  • Report Generator│
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. LangGraph Orchestrator Design

### 7.1 State Schema (`SatQueryState`)

```python
class SatQueryState(MessagesState):
    user_query: str
    uploaded_files: List[Dict]          # [{path, modality, bands, metadata}]
    intent: Optional[Dict]              # parsed task intent
    validation_result: Optional[Dict]   # {is_valid, error_message, validated_files}
    routing_decision: Optional[str]
    tool_outputs: Dict[str, Any]
    final_answer: Optional[str]
    visual_evidence: Optional[Dict]
    confidence: Optional[float]
    execution_trace: Optional[Dict]
    error: Optional[str]
```

### 7.2 Intent JSON

```json
{
  "primary_task": "vqa | captioning | grounding | change_detection | fusion | land_cover_analysis",
  "input_count": 1,
  "requires_spatial_output": false,
  "expected_modality": "optical | sar | both"
}
```

### 7.3 Graph Nodes

| Node | Purpose |
|------|---------|
| `query_parser` | NL query → Intent JSON (Llama-3.2-3B or BERT classifier) |
| `validation_gate` | Check file count, modality, format, co-registration |
| `task_router` | Conditional edge → specialist node |
| `vqa_node` | InternVL2-8B single-image VQA |
| `captioning_node` | PaliGemma-3B captioning |
| `grounding_node` | InternVL2-8B text-guided region grounding |
| `change_detection_node` | Qwen2-VL-7B bi-temporal change VQA |
| `fusion_node` | ResNet-18 sensor prior → InternVL2 with injected prompt |
| `output_combinator` | Merge outputs, confidence, execution trace |

### 7.4 Conditional Edges

- After `validation_gate`: `continue → task_router` or `error → END`
- After `task_router`: route to specialist node based on `primary_task`

### 7.5 Execution Trace Format

```json
{
  "task": "fusion",
  "models_used": ["ResNet-18", "InternVL2-8B"],
  "parameters": {"sensor_prior": "...", "top_k": 10},
  "input_count": 2,
  "timestamp": "2026-09-10T12:00:00Z",
  "duration_ms": 1234.56,
  "confidence": 0.85,
  "confidence_label": "High"
}
```

---

## 8. Hardware & Environment Constraints

| Stage | Platform | GPU | VRAM |
|-------|----------|-----|------|
| **Fine-tuning** | Google Colab Free | T4 | 16 GB |
| **Inference** | Local laptop | RTX 3050 | 4 GB |
| **Alternative** | Kaggle | T4/P100 | 16 GB |

### Memory Rules
- 4-bit quantisation (bitsandbytes) for all VLMs
- QLoRA for fine-tuning (LoRA rank 8–16)
- batch_size=1 with gradient_accumulation=8 for 7B+ models
- Gradient checkpointing for InternVL2-8B
- CPU offloading if VRAM overflows

### Colab Constraints
- Session timeout ~12 hours → checkpoint to Google Drive every 500 steps
- Monthly quota ~15–20 hours → plan experiments carefully
- Always mount Drive before training

---

## 9. ICEYE Sample Data (for SAR testing)

| Mode | Size | Resolution | Best For |
|------|------|------------|----------|
| **Dwell** | 12 MB | ~0.25m | Quick pipeline test |
| **Spot Fine** | 2 GB | ~1 m | Balanced — **recommended** |
| **Strip** | 3 GB | ~3 m | Land-cover classification |
| **Scan** | 732 MB | ~16 m | Coarse large-area |
| **Dwell Fine** | 2 GB | ~0.25m | Ultra high-res demo |

**Pairing:** For ResNet-18, pair ICEYE SAR with Sentinel-2 optical (free from Copernicus) to fill the 12-channel input.

---

## 10. Project Status

| Phase | Task | Status |
|-------|------|--------|
| 0 | Environment setup (Colab, Kaggle, dependencies) | ✅ |
| 1 | ResNet-18 integration + inference test | ✅ |
| 2 | Fine-tune PaliGemma-3B (captioning) | 🔄 |
| 3 | Fine-tune Qwen2-VL-7B (change VQA) | ⏳ |
| 4 | Fine-tune InternVL2-8B (VQA + grounding) | ⏳ |
| 5 | LangGraph agentic orchestrator | ⏳ |
| 6 | Gradio mini UI for testing | ⏳ |
| 7 | Benchmark evaluation + report | ⏳ |

---

## 11. Representative Queries (Must Handle)

| Query | Routed To |
|-------|-----------|
| "Describe the land-cover and major objects visible in this image." | `captioning_node` (PaliGemma) |
| "Highlight the water body referred to in the query." | `grounding_node` (InternVL2) |
| "What changed between these two dates, and where did the change occur?" | `change_detection_node` (Qwen2-VL) |
| "Use the optical and SAR images together to identify built-up and water-covered regions." | `fusion_node` (ResNet + InternVL2) |
| "Has the built-up area increased, decreased, or remained unchanged?" | `change_detection_node` (Qwen2-VL) |

---

## 12. Evaluation Criteria

| Component | Weight | Metric |
|-----------|--------|--------|
| Single-image VQA | 20% | Accuracy on VRSBench/RSVQA |
| Captioning / Grounding | 20% | BLEU / IoU on VRSBench |
| Change understanding | 20% | Accuracy on CDVQA |
| Optical–SAR analysis | 20% | ISRO/SAC evaluation set |
| Agentic orchestration | 20% | Correct routing + execution trace quality |

Scores are **normalised** before combining. ISRO annotations are **not disclosed** to teams.

---

## 13. File / Folder Layout

```
satquery-ai/
├── ui.py                          # Gradio interface
├── agentic_orchestrator.py        # LangGraph StateGraph
├── state_schema.py                # Pydantic SatQueryState
├── query_parser.py                # Intent extraction
├── validator.py                   # Input compatibility
├── tool_registry.py               # LangChain @tool wrappers
├── output_combinator.py           # Final answer + trace
├── audit_logger.py                # Execution logging
├── tools/
│   ├── resnet_tool.py             # ResNet-18 optical-SAR
│   ├── paligemma_tool.py          # Captioning
│   ├── internvl2_tool.py          # VQA + grounding
│   ├── qwen2vl_tool.py            # Change VQA
│   └── cdvqa_tool.py              # Change baseline
├── fusion/
│   ├── prompt_injector.py
│   └── tiling_manager.py
├── output/
│   ├── spatial_renderer.py
│   ├── confidence.py
│   └── report_generator.py
├── fine_tuning/
│   ├── paligemma/                 # JSONL + RGB extraction
│   ├── qwen2_vl/
│   └── internvl2/
├── models/                        # Downloaded weights + LoRA adapters
├── benchmarks/                    # Evaluation scripts
├── tests/
├── requirements.txt
└── context.md                     # ← this file
```

---

## 14. Dependencies

```
# Core ML
torch>=2.0.0
transformers>=4.40.0
peft>=0.10.0
bitsandbytes>=0.43.0
accelerate>=0.30.0

# Geospatial
rasterio>=1.3.0
geopandas>=0.14.0

# Agentic
langgraph>=0.2.0
langchain>=0.3.0
langchain-core>=0.3.0
langchain-openai>=0.2.0

# UI
gradio>=4.0.0

# Utils
pydantic>=2.0.0
numpy>=1.24.0
Pillow>=9.0.0
matplotlib>=3.5.0
```

---

## 15. Key Technical Decisions & Rationale

| Decision | Why |
|----------|-----|
| **ResNet-18 frozen** | Native 12-channel handler; no retraining needed; provides hard sensor prior |
| **VLMs stay RGB-only** | Avoids retraining vision tower; ResNet prior injected via prompt |
| **LangGraph over manual routing** | Observable execution trace; conditional edges; auditable state |
| **QLoRA everywhere** | Fits on T4 16 GB; only attention layers trained |
| **PaliGemma for captioning** | Lightweight; official Big Vision notebook is T4-ready |
| **Qwen2-VL for change** | Native multi-image handling; fine-tunes on CDVQA (100%) |
| **InternVL2 for VQA + grounding** | Strong high-res spatial grounding; multi-dataset mix |

---

## 16. Critical Implementation Notes

1. **BigEarthNet band extraction:** 20m bands (B05, B06, B07, B8A, B11, B12) must be upsampled to 120×120 before stacking.
2. **Prompt injection for fusion:** Build a hard system prompt: `"[Vision Tool] detects: {sensor_prior}. User asks: {query}. Answer strictly based on the image and this prior."`
3. **Execution trace is mandatory:** Every response must include task, models_used, parameters, confidence, and timing.
4. **Co-registration check:** Compare CRS and affine transform for paired images; reject mismatched inputs.
5. **PaliGemma is single-turn:** Not conversational — treat it as a stateless specialist tool.
6. **CDVQA zero-shot is poor** (~33% accuracy); fine-tuning is essential (~57% after SFT).
7. **Colab checkpointing:** Save LoRA adapters to Drive every 500 steps to survive session timeouts.
8. **ICEYE pairing:** SAR alone cannot fill ResNet's 12 channels — pair with Sentinel-2 optical.

---

## 17. Next Actions (Priority Order)

1. **Finish PaliGemma fine-tuning** on BigEarthNet caption splits → save LoRA to Drive.
2. **Build `resnet_tool.py`** with full `ResNet18Tool` class + unit tests.
3. **Fine-tune Qwen2-VL-7B** on CDVQA → save LoRA to Drive.
4. **Fine-tune InternVL2-8B** on VRSBench + RSVQA + BigEarthNet mix.
5. **Implement LangGraph orchestrator** (`agentic_orchestrator.py`).
6. **Build Gradio mini UI** (`ui.py`) with attachment upload + model selection.
7. **Run benchmark evaluations** (VRSBench, RSVQA, CDVQA) + generate results table.
8. **Prepare demo video** showing all 5 mandatory capabilities.

---

*This document is the single source of truth for SatQuery AI. Update as models are integrated and benchmark results are obtained.*
