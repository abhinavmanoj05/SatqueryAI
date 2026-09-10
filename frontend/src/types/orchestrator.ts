// ---------------------------------------------------------------------------
// OrchestratorResponse — matches the LangGraph backend JSON contract exactly.
// Update this file if the backend schema changes.
// ---------------------------------------------------------------------------

export interface BoundingBox {
  /** [x1, y1, x2, y2] in absolute pixels or normalised [0–1] coords */
  coords: [number, number, number, number]
  label?: string
  confidence?: number
}

export interface TopKClass {
  class: string
  probability: number
}

export interface ChangeStats {
  total_changed_pixels?: number
  percentage_changed?: number
  vegetation_loss_percentage?: number
  built_up_expansion_percentage?: number
  vegetation_gain_percentage?: number
  active_hotspots?: number
}

export interface VisualEvidence {
  /** Array of detected/grounded regions or bi-temporal change hotspots */
  boxes?: BoundingBox[]
  /** Base64-encoded PNG or a URL to a change mask overlay or RGBA heatmap */
  change_mask?: string
  /** Base64-encoded PNG of the satellite imagery with highlighted bounding box outlines and labels */
  annotated_image?: string
  /** Top-k land-cover class predictions */
  top_k?: TopKClass[]
  /** Bi-temporal multi-channel quantitative metrics */
  stats?: ChangeStats
  /** Multispectral semantic land-cover transition prior */
  transition?: string
}

export interface AllocationTrace {
  selection_mode: string
  router_brain: string
  allocated_models: Record<string, string>
  allocation_rationale: string
  system_telemetry: {
    ollama_status?: string
    gemini_status?: string
    physical_vision?: string
    hardware_device?: string
    [key: string]: string | undefined
  }
}

export interface ExecutionTrace {
  /** Routed task name */
  task: 'vqa' | 'captioning' | 'grounding' | 'change_detection' | 'fusion' | 'land_cover_analysis' | 'conversational'
  /** Specialist models invoked in order */
  models_used: string[]
  /** Model-specific parameters used for this run */
  parameters: Record<string, unknown>
  /** Number of input images consumed */
  input_count: number
  /** ISO-8601 timestamp of when the orchestrator started */
  timestamp: string
  /** Wall-clock time for the full pipeline in milliseconds */
  duration_ms: number
  /** Human-readable confidence label */
  confidence_label?: 'High' | 'Medium' | 'Low'
  /** Cognitive thinking / rationale of the orchestrator */
  thinking?: string
  /** Transparent 'Who Chose What and Why' allocation trace */
  allocation_trace?: AllocationTrace
}

export interface OrchestratorResponse {
  /** Primary natural-language answer from the specialist chain */
  final_answer: string
  /** Optional visual outputs (bounding boxes, change mask, probabilities) */
  visual_evidence: VisualEvidence
  /** Overall confidence score [0, 1] */
  confidence: number
  /** Full execution trace for the agent panel */
  execution_trace: ExecutionTrace
}

// ---------------------------------------------------------------------------
// Upload mode type — used by UploadZone and consumed by runSatQuery
// ---------------------------------------------------------------------------
export type UploadMode = 'single' | 'bitemporal' | 'fusion'

export interface UploadedFile {
  file: File
  preview: string   // object URL for thumbnail
  modality: 'OPTICAL' | 'SAR' | 'T0' | 'T1'
}

// ---------------------------------------------------------------------------
// Chat session types
// ---------------------------------------------------------------------------
export interface AttachedImageMeta {
  name: string
  url: string
  isTiff?: boolean
  size?: string
}

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: string
  images?: AttachedImageMeta[]
  response?: OrchestratorResponse
}

export interface Session {
  id: string
  title: string
  createdAt: string
  messages: ChatMessage[]
}
