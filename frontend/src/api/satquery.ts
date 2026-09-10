import type { OrchestratorResponse, Session } from '@/types/orchestrator'


// Public API — runSatQuery
// ---------------------------------------------------------------------------

export interface SystemModelStatus {
  ollama: {
    online: boolean
    models: Array<{
      name: string
      size?: number
      parameter_size?: string
      family?: string
      quantization?: string
    }>
    host?: string
  }
  gemini: {
    online: boolean
    models: string[]
    primary: string
  }
  physical_vision: {
    vit_base: string
    cdvqa: string
    backend: string
  }
}

export async function getAvailableModels(): Promise<SystemModelStatus | null> {
  try {
    const res = await fetch('/api/models')
    if (res.ok) {
      return (await res.json()) as SystemModelStatus
    }
  } catch (err) {
    console.warn('Failed to fetch /api/models:', err)
  }
  return null
}

/**
 * Run a live SatQuery AI analysis request against the FastAPI agentic backend.
 * Never falls back to dummy stubs.
 */
export async function runSatQuery(
  query: string,
  files: File[],
  apiKey?: string,
  preferredModel?: string,
): Promise<OrchestratorResponse> {
  const formData = new FormData()
  formData.append('query', query)
  if (apiKey) {
    formData.append('api_key', apiKey)
  }
  if (preferredModel) {
    formData.append('preferred_model', preferredModel)
  }
  files.forEach((f) => formData.append('files', f))

  let res: Response
  try {
    res = await fetch('/api/analyze', {
      method: 'POST',
      body: formData,
    })
  } catch (err) {
    throw new Error(
      `Failed to reach SatQuery AI backend service (http://127.0.0.1:8000). Ensure the backend is running. Details: ${err instanceof Error ? err.message : String(err)}`
    )
  }

  if (!res.ok) {
    let errorDetail = `HTTP ${res.status}: ${res.statusText}`
    try {
      const errJson = await res.json()
      if (errJson.detail) {
        errorDetail = typeof errJson.detail === 'string' ? errJson.detail : JSON.stringify(errJson.detail)
      }
    } catch {
      try {
        const text = await res.text()
        if (text) errorDetail = text.slice(0, 300)
      } catch {}
    }
    throw new Error(`SatQuery Backend Error: ${errorDetail}`)
  }

  const data = (await res.json()) as OrchestratorResponse
  if (!data || !data.final_answer) {
    throw new Error('Received malformed response payload from backend.')
  }

  return data
}

/**
 * Generate a synthesized Earth Observation Research Report summarizing the active session.
 */
export async function generateSessionReport(
  session: Session,
  preferredModel?: string,
  apiKey?: string,
): Promise<{ executiveSummary: string; modelUsed: string }> {
  try {
    const payload = {
      session_id: session.id,
      session_title: session.title,
      messages: session.messages.map((m) => ({
        role: m.role,
        content: m.content,
        timestamp: m.timestamp,
        images: m.images?.map((img) => ({ name: img.name, size: img.size, isTiff: img.isTiff })),
        response: m.response
          ? {
              final_answer: m.response.final_answer,
              visual_evidence: m.response.visual_evidence,
              confidence: m.response.confidence,
              execution_trace: m.response.execution_trace,
            }
          : undefined,
      })),
      preferred_model: preferredModel || 'auto',
      api_key: apiKey,
    }

    const res = await fetch('/api/generate-report', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })

    if (res.ok) {
      const data = await res.json()
      return {
        executiveSummary: data.executive_summary || data.executiveSummary || '',
        modelUsed: data.model_used || data.modelUsed || 'SatQuery Intelligence Engine',
      }
    }
  } catch (err) {
    console.warn('Backend report generation endpoint unreachable, falling back to client synthesizer:', err)
  }

  throw new Error('Failed to generate AI executive summary from backend')
}
