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

export interface VisualEvidence {
  /** Array of detected/grounded regions */
  boxes?: BoundingBox[]
  /** Base64-encoded PNG or a URL to a change mask overlay */
  change_mask?: string
  /** Top-k land-cover class predictions */
  top_k?: TopKClass[]
}

export interface ExecutionTrace {
  /** Routed task name */
  task: 'vqa' | 'captioning' | 'grounding' | 'change_detection' | 'fusion' | 'land_cover_analysis'
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
export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: string
  response?: OrchestratorResponse
}

export interface Session {
  id: string
  title: string
  createdAt: string
  messages: ChatMessage[]
}
