import { ShieldCheck, Crosshair, BarChart2, Layers, CheckCircle2 } from 'lucide-react'
import type { ChatMessage, OrchestratorResponse } from '@/types/orchestrator'

interface ResponsePanelProps {
  messages: ChatMessage[]
  isLoading: boolean
}

export default function ResponsePanel({ messages, isLoading }: ResponsePanelProps) {
  if (messages.length === 0 && !isLoading) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center p-8 text-center bg-navy-50/20">
        <div className="w-12 h-12 rounded-2xl bg-sky-50 border border-sky-100 flex items-center justify-center text-saffron mb-3">
          <Layers className="w-6 h-6" />
        </div>
        <h3 className="text-base font-semibold text-navy mb-1">SatQuery AI Workspace</h3>
        <p className="text-xs text-navy-500 max-w-md leading-relaxed">
          Upload satellite imagery in the panel above and type your analysis query below.
          The LangGraph agent will dynamically select specialist models and ground answers with visual evidence.
        </p>
      </div>
    )
  }

  return (
    <div className="flex-1 overflow-y-auto p-4 space-y-6">
      {messages.map((msg) => (
        <div key={msg.id} className="space-y-4">
          {/* User Message */}
          {msg.role === 'user' && (
            <div className="flex justify-end">
              <div className="max-w-2xl bg-navy-800 text-white rounded-2xl rounded-tr-sm px-4 py-3 text-sm shadow-sm">
                <p className="leading-relaxed">{msg.content}</p>
                <div className="text-[10px] text-navy-300 mt-1 text-right">{msg.timestamp}</div>
              </div>
            </div>
          )}

          {/* Assistant Response */}
          {msg.role === 'assistant' && msg.response && (
            <AssistantResponseCard response={msg.response} timestamp={msg.timestamp} />
          )}
        </div>
      ))}

      {/* Loading Skeleton */}
      {isLoading && (
        <div className="bg-white rounded-2xl border border-navy-100 p-5 shadow-card space-y-4 animate-pulse">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-navy-200 rounded-full" />
            <div className="h-4 w-32 bg-navy-200 rounded" />
          </div>
          <div className="space-y-2">
            <div className="h-3.5 bg-navy-100 rounded w-full" />
            <div className="h-3.5 bg-navy-100 rounded w-5/6" />
            <div className="h-3.5 bg-navy-100 rounded w-3/4" />
          </div>
          <div className="h-28 bg-navy-50 rounded-xl border border-navy-100" />
        </div>
      )}
    </div>
  )
}

function AssistantResponseCard({
  response,
  timestamp,
}: {
  response: OrchestratorResponse
  timestamp: string
}) {
  const { final_answer, visual_evidence, confidence, execution_trace } = response

  const confidencePct = Math.round(confidence * 100)
  const confidenceColor =
    confidence >= 0.8
      ? 'text-emerald-700 bg-emerald-50 border-emerald-200'
      : confidence >= 0.6
      ? 'text-amber-700 bg-amber-50 border-amber-200'
      : 'text-red-700 bg-red-50 border-red-200'

  return (
    <div className="bg-white rounded-2xl border border-navy-100 p-5 shadow-card space-y-5">
      {/* Card Header: Confidence & Models */}
      <div className="flex items-center justify-between flex-wrap gap-2 pb-3 border-b border-navy-100">
        <div className="flex items-center gap-2">
          <div className="w-6 h-6 rounded-md bg-navy-50 flex items-center justify-center text-saffron">
            <CheckCircle2 className="w-4 h-4" />
          </div>
          <span className="text-xs font-semibold text-navy uppercase tracking-wider">
            Agent Intelligence Response
          </span>
        </div>

        <div className="flex items-center gap-2">
          {/* Confidence Badge */}
          <div
            className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold border ${confidenceColor}`}
          >
            <ShieldCheck className="w-3.5 h-3.5" />
            <span>Confidence: {confidencePct}%</span>
          </div>

          <span className="text-[11px] text-navy-400 font-mono">{timestamp}</span>
        </div>
      </div>

      {/* Answer Body */}
      <div className="text-sm text-navy-800 leading-relaxed space-y-3 whitespace-pre-line font-normal">
        {final_answer}
      </div>

      {/* Visual Evidence Section */}
      {(visual_evidence.top_k || visual_evidence.boxes || visual_evidence.change_mask) && (
        <div className="bg-navy-50/50 rounded-xl p-4 border border-navy-100 space-y-4">
          <div className="flex items-center gap-2 text-xs font-bold text-navy uppercase tracking-wider">
            <BarChart2 className="w-4 h-4 text-saffron" />
            <span>Visual Evidence & Spectral Signatures</span>
          </div>

          {/* Top-K Probability Bars */}
          {visual_evidence.top_k && visual_evidence.top_k.length > 0 && (
            <div className="space-y-2.5">
              <span className="text-xs font-medium text-navy-600 block">
                CORINE Land Cover Predictions
              </span>
              <div className="space-y-2">
                {visual_evidence.top_k.map((item, idx) => {
                  const prob = Math.round(item.probability * 100)
                  return (
                    <div key={idx} className="space-y-1">
                      <div className="flex justify-between text-xs text-navy-700">
                        <span className="font-medium">{item.class}</span>
                        <span className="font-mono text-navy-500">{prob}%</span>
                      </div>
                      <div className="w-full bg-navy-100 rounded-full h-2 overflow-hidden">
                        <div
                          className="bg-navy-700 h-2 rounded-full transition-all duration-500"
                          style={{ width: `${prob}%` }}
                        />
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>
          )}

          {/* Bounding Box Visual Evidence */}
          {visual_evidence.boxes && visual_evidence.boxes.length > 0 && (
            <div className="space-y-2 pt-2 border-t border-navy-100">
              <div className="flex items-center gap-1.5 text-xs font-semibold text-navy-700">
                <Crosshair className="w-3.5 h-3.5 text-saffron" />
                <span>Localized Bounding Boxes ({visual_evidence.boxes.length} targets)</span>
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                {visual_evidence.boxes.map((box, i) => (
                  <div
                    key={i}
                    className="p-2.5 bg-white rounded-lg border border-navy-100 text-xs text-navy-700 flex flex-col gap-1"
                  >
                    <div className="flex justify-between font-medium">
                      <span>{box.label || `Region ${i + 1}`}</span>
                      {box.confidence && (
                        <span className="text-[10px] text-emerald-600 font-mono">
                          {Math.round(box.confidence * 100)}%
                        </span>
                      )}
                    </div>
                    <code className="text-[11px] font-mono text-navy-500 bg-navy-50 px-1.5 py-0.5 rounded">
                      [{box.coords.join(', ')}]
                    </code>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Change Mask Evidence */}
          {visual_evidence.change_mask && (
            <div className="space-y-2 pt-2 border-t border-navy-100">
              <span className="text-xs font-semibold text-navy-700 block">
                Temporal Change Delta Mask
              </span>
              <div className="p-2 bg-white rounded-lg border border-navy-100 inline-block">
                <img
                  src={visual_evidence.change_mask}
                  alt="Detected change mask overlay"
                  className="max-h-40 rounded border border-navy-100 object-contain bg-navy-900"
                />
              </div>
            </div>
          )}
        </div>
      )}

      {/* Model Footnote */}
      <div className="text-[11px] text-navy-400 flex items-center justify-between pt-1">
        <span>Models: {execution_trace.models_used.join(' → ')}</span>
        <span>Duration: {execution_trace.duration_ms.toFixed(0)} ms</span>
      </div>
    </div>
  )
}
