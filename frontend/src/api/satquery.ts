import type { OrchestratorResponse } from '@/types/orchestrator'


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
