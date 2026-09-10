import { useState } from 'react'
import {
  ShieldCheck,
  Crosshair,
  BarChart2,
  Layers,
  CheckCircle2,
  Brain,
  ChevronDown,
  ChevronUp,
  Sparkles,
  Activity,
  Server,
  MapPin,
} from 'lucide-react'
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
          The cognitive LangGraph orchestrator will reason about your request, select specialist models, and ground answers with visual evidence.
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
          {msg.role === 'assistant' && (
            <AssistantResponseCard
              response={msg.response}
              fallbackContent={msg.content}
              timestamp={msg.timestamp}
            />
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
  fallbackContent,
  timestamp,
}: {
  response?: OrchestratorResponse
  fallbackContent?: string
  timestamp: string
}) {
  const [showThinking, setShowThinking] = useState(true)
  const [showAllocation, setShowAllocation] = useState(true)

  if (!response) {
    return (
      <div className="bg-white rounded-2xl border border-navy-100 p-5 shadow-card space-y-3">
        <div className="flex items-center gap-2 text-xs font-semibold text-navy uppercase">
          <CheckCircle2 className="w-4 h-4 text-saffron" />
          <span>Agent Intelligence Response</span>
        </div>
        <div className="text-sm text-navy-800 leading-relaxed whitespace-pre-line">
          {fallbackContent || 'No response content available.'}
        </div>
        <div className="text-[10px] text-navy-400 font-mono">{timestamp}</div>
      </div>
    )
  }

  const { final_answer, visual_evidence, confidence = 0.8, execution_trace } = response
  const evidence = visual_evidence || {}

  const confidencePct = Math.round(confidence * 100)
  const confidenceColor =
    confidence >= 0.8
      ? 'text-emerald-700 bg-emerald-50 border-emerald-200'
      : confidence >= 0.6
      ? 'text-amber-700 bg-amber-50 border-amber-200'
      : 'text-red-700 bg-red-50 border-red-200'

  const hasVisualEvidence = Boolean(
    (evidence.top_k && evidence.top_k.length > 0) ||
    (evidence.boxes && evidence.boxes.length > 0) ||
    evidence.change_mask ||
    evidence.stats ||
    evidence.transition
  )

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

      {/* "Who Chose What and All" Omni-Route Model Allocation Banner */}
      {execution_trace?.allocation_trace && (
        <div className="bg-gradient-to-r from-navy-900 to-navy-800 text-white rounded-xl p-4 shadow-sm border border-navy-700 space-y-3">
          <div className="flex items-center justify-between flex-wrap gap-2">
            <div className="flex items-center gap-2">
              <div className="p-1 rounded bg-saffron/20 text-saffron">
                <Sparkles className="w-3.5 h-3.5" />
              </div>
              <span className="text-xs font-bold uppercase tracking-wider text-saffron">
                Omni-Route Model Allocation
              </span>
              <span className="text-navy-400 text-xs">|</span>
              <span className="text-xs text-navy-200 font-semibold">
                {execution_trace.allocation_trace.selection_mode}
              </span>
            </div>
            <button
              type="button"
              onClick={() => setShowAllocation(!showAllocation)}
              className="text-navy-300 hover:text-white transition-colors"
            >
              {showAllocation ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
            </button>
          </div>

          {showAllocation && (
            <div className="space-y-3 pt-2 border-t border-navy-700/60 text-xs">
              {/* Who Chose What Summary */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-[11px]">
                <div className="bg-navy-950/60 p-2.5 rounded border border-navy-800">
                  <span className="text-navy-400 block font-sans mb-0.5">Decision Router:</span>
                  <span className="font-semibold text-emerald-400 font-mono">
                    {execution_trace.allocation_trace.router_brain}
                  </span>
                </div>
                <div className="bg-navy-950/60 p-2.5 rounded border border-navy-800">
                  <span className="text-navy-400 block font-sans mb-0.5">Specialist Pipeline:</span>
                  <span className="font-semibold text-sky-300 font-mono">
                    {Object.values(execution_trace.allocation_trace.allocated_models).join(' + ')}
                  </span>
                </div>
              </div>

              {/* Allocation Rationale */}
              <div className="bg-navy-950/40 p-2.5 rounded border border-navy-800/80 text-[11px] text-navy-200 leading-relaxed font-sans">
                <span className="font-bold text-saffron block mb-0.5">Allocation Rationale:</span>
                {execution_trace.allocation_trace.allocation_rationale}
              </div>

              {/* System Telemetry Chips */}
              {execution_trace.allocation_trace.system_telemetry && (
                <div className="flex items-center gap-2 flex-wrap text-[10px] font-mono text-navy-300 pt-1">
                  <span className="bg-navy-800 px-2 py-0.5 rounded border border-navy-700">
                    Ollama: {execution_trace.allocation_trace.system_telemetry.ollama_status}
                  </span>
                  <span className="bg-navy-800 px-2 py-0.5 rounded border border-navy-700">
                    Gemini: {execution_trace.allocation_trace.system_telemetry.gemini_status}
                  </span>
                  <span className="bg-navy-800 px-2 py-0.5 rounded border border-navy-700">
                    Target: {execution_trace.allocation_trace.system_telemetry.physical_vision}
                  </span>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Orchestrator Cognitive Thinking Rationale */}
      {execution_trace?.thinking && (
        <div className="bg-navy-50/80 rounded-xl p-3.5 border border-navy-200/70 space-y-2">
          <button
            type="button"
            onClick={() => setShowThinking(!showThinking)}
            className="w-full flex items-center justify-between text-left"
          >
            <div className="flex items-center gap-1.5 text-xs font-semibold text-navy-800 uppercase tracking-wider">
              <Brain className="w-3.5 h-3.5 text-saffron" />
              <span>Orchestrator Cognitive Rationale & Thinking</span>
            </div>
            {showThinking ? (
              <ChevronUp className="w-4 h-4 text-navy-400" />
            ) : (
              <ChevronDown className="w-4 h-4 text-navy-400" />
            )}
          </button>
          {showThinking && (
            <p className="text-xs text-navy-700 leading-relaxed font-sans pt-1 border-t border-navy-200/50">
              {execution_trace.thinking}
            </p>
          )}
        </div>
      )}

      {/* Answer Body */}
      <div className="text-sm text-navy-800 leading-relaxed space-y-3 whitespace-pre-line font-normal">
        {final_answer || fallbackContent}
      </div>

      {/* Visual Evidence Section */}
      {hasVisualEvidence && (
        <div className="bg-navy-50/50 rounded-xl p-4 border border-navy-100 space-y-4">
          <div className="flex items-center gap-2 text-xs font-bold text-navy uppercase tracking-wider">
            <BarChart2 className="w-4 h-4 text-saffron" />
            <span>Visual Evidence & Spectral Signatures</span>
          </div>

          {/* Top-K Probability Bars */}
          {evidence.top_k && evidence.top_k.length > 0 && (
            <div className="space-y-2.5">
              <span className="text-xs font-medium text-navy-600 block">
                CORINE Land Cover Predictions
              </span>
              <div className="space-y-2">
                {evidence.top_k.map((item, idx) => {
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
          {evidence.boxes && evidence.boxes.length > 0 && (
            <div className="space-y-2 pt-2 border-t border-navy-100">
              <div className="flex items-center gap-1.5 text-xs font-semibold text-navy-700">
                <Crosshair className="w-3.5 h-3.5 text-saffron" />
                <span>Localized Bounding Boxes ({evidence.boxes.length} targets)</span>
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                {evidence.boxes.map((box, i) => (
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

          {/* Bi-Temporal Quantitative Analytics & Land-Cover Transition */}
          {(evidence.stats || evidence.transition) && (
            <div className="space-y-3 pt-3 border-t border-navy-100">
              <div className="flex items-center justify-between flex-wrap gap-2">
                <div className="flex items-center gap-1.5 text-xs font-bold text-navy uppercase tracking-wider">
                  <Activity className="w-3.5 h-3.5 text-saffron" />
                  <span>Bi-Temporal Surface Analytics & Transition</span>
                </div>
                {evidence.stats?.percentage_changed !== undefined && (
                  <span className="text-xs font-mono font-bold bg-amber-100 text-amber-900 px-2.5 py-0.5 rounded-full border border-amber-200">
                    Δ {evidence.stats.percentage_changed}% Total Scene Variation
                  </span>
                )}
              </div>

              {/* Land-Cover Transition Pill */}
              {evidence.transition && (
                <div className="p-3 bg-sky-50/80 rounded-lg border border-sky-100 text-xs font-sans text-navy-800 space-y-1">
                  <span className="font-bold text-navy-900 block text-xs">
                    Dual ViT-Base Land-Cover Transition Matrix:
                  </span>
                  <div className="whitespace-pre-line text-[11px] leading-relaxed font-mono bg-white p-2.5 rounded border border-sky-200/80 text-navy-700">
                    {evidence.transition}
                  </div>
                </div>
              )}

              {/* Quantitative Metrics Grid */}
              {evidence.stats && (
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs">
                  <div className="p-2.5 bg-red-50/80 rounded-lg border border-red-200 text-center">
                    <span className="text-[10px] text-red-600 block uppercase font-semibold">Vegetation Clearing</span>
                    <span className="text-base font-extrabold text-red-800 font-mono">
                      -{evidence.stats.vegetation_loss_percentage ?? 0}%
                    </span>
                  </div>
                  <div className="p-2.5 bg-amber-50/80 rounded-lg border border-amber-200 text-center">
                    <span className="text-[10px] text-amber-700 block uppercase font-semibold">Built-Up Expansion</span>
                    <span className="text-base font-extrabold text-amber-900 font-mono">
                      +{evidence.stats.built_up_expansion_percentage ?? 0}%
                    </span>
                  </div>
                  <div className="p-2.5 bg-emerald-50/80 rounded-lg border border-emerald-200 text-center">
                    <span className="text-[10px] text-emerald-700 block uppercase font-semibold">Revegetation</span>
                    <span className="text-base font-extrabold text-emerald-900 font-mono">
                      +{evidence.stats.vegetation_gain_percentage ?? 0}%
                    </span>
                  </div>
                  <div className="p-2.5 bg-navy-50/80 rounded-lg border border-navy-200 text-center">
                    <span className="text-[10px] text-navy-600 block uppercase font-semibold">Active Hotspots</span>
                    <span className="text-base font-extrabold text-navy-900 font-mono">
                      {evidence.stats.active_hotspots ?? evidence.boxes?.length ?? 0}
                    </span>
                  </div>
                </div>
              )}

              {/* Change Heatmap with Color Legend */}
              {evidence.change_mask && (
                <div className="space-y-2 pt-1">
                  <span className="text-xs font-semibold text-navy-700 block">
                    Spectral Delta Heatmap & Hotspot Overlay:
                  </span>
                  <div className="flex flex-col sm:flex-row items-start gap-4 p-3 bg-white rounded-lg border border-navy-100">
                    <div className="p-1.5 bg-navy-950 rounded-lg border border-navy-800 shadow-inner">
                      <img
                        src={evidence.change_mask}
                        alt="Bi-Temporal Change Heatmap"
                        className="max-h-44 rounded object-contain"
                      />
                    </div>
                    <div className="space-y-2 text-[11px] text-navy-600">
                      <span className="font-bold text-navy-800 block text-xs">Heatmap Classification Legend:</span>
                      <div className="flex items-center gap-2">
                        <span className="w-3 h-3 rounded-full bg-red-600 inline-block flex-shrink-0" />
                        <span>🟥 Red: Vegetation clearing / loss (ΔVeg &lt; -8%)</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="w-3 h-3 rounded-full bg-yellow-400 inline-block flex-shrink-0" />
                        <span>🟨 Yellow: Built-up / new reflective structures (ΔBright &gt; +10%)</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="w-3 h-3 rounded-full bg-emerald-500 inline-block flex-shrink-0" />
                        <span>🟩 Green: Revegetation / agricultural growth (ΔVeg &gt; +8%)</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="w-3 h-3 rounded-full bg-orange-500 inline-block flex-shrink-0" />
                        <span>🟧 Orange: Soil & surface matrix variation</span>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Simple Change Mask Evidence fallback if stats not available */}
          {!evidence.stats && evidence.change_mask && (
            <div className="space-y-2 pt-2 border-t border-navy-100">
              <span className="text-xs font-semibold text-navy-700 block">
                Temporal Change Delta Mask
              </span>
              <div className="p-2 bg-white rounded-lg border border-navy-100 inline-block">
                <img
                  src={evidence.change_mask}
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
        <span>Models: {execution_trace?.models_used?.join(' → ') || 'Orchestrator'}</span>
        <span>Duration: {execution_trace?.duration_ms ? execution_trace.duration_ms.toFixed(0) : '0'} ms</span>
      </div>
    </div>
  )
}
