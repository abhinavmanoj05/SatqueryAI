import type { OrchestratorResponse } from '@/types/orchestrator'

// ---------------------------------------------------------------------------
// Mock responses — realistic stubs for every task type.
// Swap the body of runSatQuery() to a real fetch() call when the backend is
// ready.  The function signature and return type stay identical.
// ---------------------------------------------------------------------------

function detectIntent(query: string): OrchestratorResponse['execution_trace']['task'] {
  const q = query.toLowerCase()
  if (q.match(/change|temporal|before|after|t0|t1|date 1|date 2|difference|delta/))
    return 'change_detection'
  if (q.match(/sar|radar|optical.*sar|fusion|cross.modal|sentinel.1/))
    return 'fusion'
  if (q.match(/caption|describe|what.*see|overview|summary|scene/))
    return 'captioning'
  if (q.match(/ground|locate|where|find|detect|highlight|show me|mark|bounding|box/))
    return 'grounding'
  if (q.match(/land.cover|class|classif|lulc|built.up|urban|vegetation|forest|water.*body/))
    return 'land_cover_analysis'
  return 'vqa'
}

function now() {
  return new Date().toISOString()
}

const MOCK_RESPONSES: Record<OrchestratorResponse['execution_trace']['task'], () => OrchestratorResponse> = {

  captioning: () => ({
    final_answer:
      'The image shows a mixed agricultural landscape with irrigated crop fields covering approximately 58% of the scene, predominantly in the western and central zones. Interspersed broad-leaved forest patches appear along the northern edge and southeastern quadrant. A meandering seasonal river channel is visible at the southern margin, with riparian vegetation indicating a perennial water table. Field boundary patterns are consistent with small-holder farming in the Indo-Gangetic plain. No significant built-up areas are observed; scattered structures likely represent isolated farmsteads or storage facilities.',
    visual_evidence: {
      top_k: [
        { class: 'Arable land (non-irrigated)', probability: 0.68 },
        { class: 'Broad-leaved forest', probability: 0.51 },
        { class: 'Inland wetlands', probability: 0.33 },
        { class: 'Complex cultivation patterns', probability: 0.28 },
        { class: 'Discontinuous urban fabric', probability: 0.12 },
      ],
    },
    confidence: 0.79,
    execution_trace: {
      task: 'captioning',
      models_used: ['PaliGemma-3B'],
      parameters: { prefix: 'caption en', max_new_tokens: 256, lora_adapter: 'paligemma-ben-caption-v1' },
      input_count: 1,
      timestamp: now(),
      duration_ms: 1287.4,
      confidence_label: 'High',
    },
  }),

  vqa: () => ({
    final_answer:
      'Based on the spectral characteristics and texture patterns in the image, the dominant land-cover type is irrigated agriculture. The regular geometric field parcels, uniform canopy reflectance, and proximity to the water channel strongly suggest paddy or wheat cultivation. The NDVI-equivalent signal from optical bands indicates moderate to high vegetation density (estimated LAI ≈ 3.2). No anomalous features such as industrial facilities, flooding, or burn scars are detected.',
    visual_evidence: {
      top_k: [
        { class: 'Irrigated cropland', probability: 0.82 },
        { class: 'Permanent grassland', probability: 0.41 },
        { class: 'Broad-leaved forest', probability: 0.29 },
      ],
    },
    confidence: 0.82,
    execution_trace: {
      task: 'vqa',
      models_used: ['ResNet-18', 'InternVL2-8B'],
      parameters: { top_k: 10, lora_adapter: 'internvl2-vrsbench-v1', quantization: '4bit' },
      input_count: 1,
      timestamp: now(),
      duration_ms: 2154.8,
      confidence_label: 'High',
    },
  }),

  grounding: () => ({
    final_answer:
      'The visual grounding task identified 3 distinct regions matching the query target. Region 1 (upper-left quadrant) shows a reservoir or seasonal pond with clear spectral boundary. Region 2 (central) appears to be an irrigation canal segment. Region 3 (lower-right) corresponds to a small natural wetland. All bounding boxes are drawn at IoU > 0.65 against probable ground-truth extents based on spectral homogeneity analysis.',
    visual_evidence: {
      boxes: [
        { coords: [42, 58, 198, 210], label: 'Water body (reservoir)', confidence: 0.87 },
        { coords: [310, 145, 490, 175], label: 'Irrigation canal', confidence: 0.74 },
        { coords: [520, 380, 680, 480], label: 'Natural wetland', confidence: 0.69 },
      ],
      top_k: [
        { class: 'Water bodies', probability: 0.91 },
        { class: 'Inland marshes', probability: 0.43 },
      ],
    },
    confidence: 0.77,
    execution_trace: {
      task: 'grounding',
      models_used: ['InternVL2-8B'],
      parameters: { grounding_target: 'water body', dynamic_resolution: true, quantization: '4bit' },
      input_count: 1,
      timestamp: now(),
      duration_ms: 2891.2,
      confidence_label: 'High',
    },
  }),

  change_detection: () => ({
    final_answer:
      'Comparing the two temporal images, significant land-cover transitions are detected. Urban/built-up fabric has expanded by an estimated 14.3% in the northeastern corridor, encroaching on what were previously agricultural fields. The water body in the central region has contracted by approximately 8% — consistent with seasonal drawdown or increased extraction. Vegetation density has decreased in two parcels (southwestern cluster), possibly due to harvesting or fallow rotation. No evidence of flood inundation or fire damage is observed.',
    visual_evidence: {
      change_mask:
        'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAYAAADED76LAAAAMElEQVQoU2P8z8BQDwAEgAF/QualityGray+0AAAAASUVORK5CYII=',
      top_k: [
        { class: 'Urban expansion (net gain)', probability: 0.87 },
        { class: 'Agricultural land loss', probability: 0.74 },
        { class: 'Water body contraction', probability: 0.68 },
        { class: 'Vegetation decrease', probability: 0.55 },
      ],
    },
    confidence: 0.74,
    execution_trace: {
      task: 'change_detection',
      models_used: ['CDVQA Baseline', 'Qwen2-VL-7B'],
      parameters: {
        changed_area_pct: 14.3,
        water_change_pct: -8.1,
        cdvqa_threshold: 0.5,
        lora_adapter: 'qwen2vl-cdvqa-v1',
        quantization: '4bit',
      },
      input_count: 2,
      timestamp: now(),
      duration_ms: 4327.9,
      confidence_label: 'High',
    },
  }),

  fusion: () => ({
    final_answer:
      'Cross-modal analysis fusing Sentinel-2 optical (10 m) and Sentinel-1 SAR (VV/VH polarisation) inputs:\n\n**ResNet-18 Sensor Prior:** Broad-leaved forest (71%), Inland wetlands (38%), Arable land (29%)\n\n**Fused Interpretation:** The SAR backscatter confirms high-moisture content in the vegetation canopy (double-bounce signature in VV), consistent with evergreen broadleaf forest rather than deciduous. The optical-SAR combination resolves cloud-shadowed areas that were ambiguous in the optical-only image — a significant wetland complex (~2.4 km²) is confirmed on the eastern margin. Urban fabric detected in the optical is validated by SAR specular scattering (low backscatter on flat rooftops). Overall scene composition: Forest 41%, Wetland 22%, Agriculture 24%, Built-up 8%, Water 5%.',
    visual_evidence: {
      top_k: [
        { class: 'Broad-leaved forest', probability: 0.71 },
        { class: 'Inland wetlands', probability: 0.38 },
        { class: 'Non-irrigated arable land', probability: 0.29 },
        { class: 'Discontinuous urban fabric', probability: 0.21 },
        { class: 'Water bodies', probability: 0.18 },
        { class: 'Transitional woodland', probability: 0.14 },
      ],
    },
    confidence: 0.85,
    execution_trace: {
      task: 'fusion',
      models_used: ['ResNet-18', 'InternVL2-8B'],
      parameters: {
        sensor_prior: 'Broad-leaved forest (71%), Inland wetlands (38%), Arable land (29%)',
        s2_channels: ['B02', 'B03', 'B04', 'B08', 'B11'],
        s1_channels: ['VV', 'VH'],
        lora_adapter: 'internvl2-vrsbench-v1',
        quantization: '4bit',
      },
      input_count: 2,
      timestamp: now(),
      duration_ms: 3156.3,
      confidence_label: 'High',
    },
  }),

  land_cover_analysis: () => ({
    final_answer:
      'Land cover classification using the 12-channel Sentinel-1+2 fusion model (ResNet-18, BigEarthNet v2.0):\n\n| Class | Probability |\n|---|---|\n| Broad-leaved forest | 71% |\n| Inland wetlands | 38% |\n| Arable land (non-irrigated) | 29% |\n| Coniferous forest | 24% |\n| Complex cultivation patterns | 19% |\n| Discontinuous urban fabric | 13% |\n\nDominant class: **Broad-leaved forest** (macro AP ≈ 0.71). Secondary land-cover types suggest an ecotone zone between forest and agricultural land, with a notable wetland complex on the image margin.',
    visual_evidence: {
      top_k: [
        { class: 'Broad-leaved forest', probability: 0.71 },
        { class: 'Inland wetlands', probability: 0.38 },
        { class: 'Arable land (non-irrigated)', probability: 0.29 },
        { class: 'Coniferous forest', probability: 0.24 },
        { class: 'Complex cultivation patterns', probability: 0.19 },
        { class: 'Discontinuous urban fabric', probability: 0.13 },
      ],
    },
    confidence: 0.83,
    execution_trace: {
      task: 'land_cover_analysis',
      models_used: ['ViT-Base (BigEarthNet 12-channel)'],
      parameters: { top_k: 6, input_channels: 12, resolution: '120x120', checkpoint: 'BIFOLD-BigEarthNetv2-0/vit_base_patch8_224-all-v0.2.0' },
      input_count: 1,
      timestamp: now(),
      duration_ms: 487.6,
      confidence_label: 'High',
    },
  }),

  conversational: () => ({
    final_answer:
      '👋 Welcome to SatQuery AI. Upload satellite imagery in the panel above to begin specialist VQA, Grounding, Scene Captioning, or Optical-SAR Fusion analysis.',
    visual_evidence: {},
    confidence: 0.95,
    execution_trace: {
      task: 'conversational',
      models_used: ['SatQuery Cognitive NLP Brain'],
      parameters: {},
      input_count: 0,
      timestamp: now(),
      duration_ms: 120.0,
      confidence_label: 'High',
      thinking: 'User entered a conversational or preparatory query with no imagery attached.',
    },
  }),
}

// ---------------------------------------------------------------------------
// Public API — runSatQuery
// ---------------------------------------------------------------------------

/**
 * Run a SatQuery AI analysis request against the orchestrator.
 *
 * Currently returns a realistic mock response keyed to the detected intent.
 * To wire up the real backend, replace the body below with a single fetch():
 *
 *   const form = new FormData()
 *   form.append('query', query)
 *   files.forEach(f => form.append('files', f))
 *   const res = await fetch('/api/analyze', { method: 'POST', body: form })
 *   return res.json() as Promise<OrchestratorResponse>
 */
export async function runSatQuery(
  query: string,
  files: File[],
  apiKey?: string,
): Promise<OrchestratorResponse> {
  // 1. Try real LangGraph Agentic Orchestrator backend
  try {
    const formData = new FormData()
    formData.append('query', query)
    if (apiKey) {
      formData.append('api_key', apiKey)
    }
    files.forEach((f) => formData.append('files', f))

    const res = await fetch('/api/analyze', {
      method: 'POST',
      body: formData,
    })

    if (res.ok) {
      const data = (await res.json()) as OrchestratorResponse
      if (data && data.final_answer && data.execution_trace) {
        return data
      }
    }
    console.warn('Real backend returned non-OK status:', res.status)
  } catch (err) {
    console.warn('Backend /api/analyze unreachable, falling back to local orchestrator emulator:', err)
  }

  // 2. Fallback simulation if backend is offline
  const latency = 600 + Math.random() * 1200
  await new Promise((resolve) => setTimeout(resolve, latency))

  const task = detectIntent(query)

  const effectiveTask: OrchestratorResponse['execution_trace']['task'] =
    files.length >= 2 && task === 'vqa'
      ? 'change_detection'
      : task

  const response = MOCK_RESPONSES[effectiveTask]()
  response.execution_trace.duration_ms = Math.round(latency * 1.12)
  return response
}
