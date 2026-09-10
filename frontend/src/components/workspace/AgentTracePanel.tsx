import { Terminal, Activity, ChevronLeft, ChevronRight, CheckCircle2, Clock, Brain } from 'lucide-react'
import type { ExecutionTrace } from '@/types/orchestrator'

interface AgentTracePanelProps {
  trace?: ExecutionTrace
  isOpen: boolean
  onToggle: () => void
}

export default function AgentTracePanel({ trace, isOpen, onToggle }: AgentTracePanelProps) {
  return (
    <aside
      className={`relative transition-all duration-300 ease-in-out border-l border-navy-100 bg-white flex flex-col z-20 ${
        isOpen ? 'w-80' : 'w-12'
      }`}
    >
      {/* Header */}
      <div className="p-3 border-b border-navy-100 flex items-center justify-between">
        <button
          onClick={onToggle}
          className="p-1 rounded-md text-navy-400 hover:text-navy hover:bg-navy-50 transition-colors"
          title={isOpen ? 'Collapse Agent Panel' : 'Expand Agent Panel'}
          aria-label={isOpen ? 'Collapse Agent Panel' : 'Expand Agent Panel'}
        >
          {isOpen ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
        </button>

        {isOpen && (
          <div className="flex items-center gap-1.5 text-xs font-mono font-semibold text-navy uppercase tracking-wider">
            <Terminal className="w-4 h-4 text-saffron" />
            <span>Agent Execution Trace</span>
          </div>
        )}
      </div>

      {/* Content */}
      {isOpen ? (
        <div className="flex-1 overflow-y-auto p-4 space-y-4 font-mono text-xs text-navy-800">
          {trace ? (
            <>
              {/* System State Banner */}
              <div className="p-3 bg-navy-50 rounded-lg border border-navy-100 space-y-2">
                <div className="flex items-center justify-between text-[11px]">
                  <span className="text-navy-500 uppercase font-semibold">Orchestrator State</span>
                  <span className="flex items-center gap-1 text-emerald-600 font-bold">
                    <CheckCircle2 className="w-3.5 h-3.5" /> Complete
                  </span>
                </div>
                <div className="text-[11px] text-navy-700">
                  <span className="text-navy-400">Task:</span> <span className="font-bold text-saffron">{trace.task}</span>
                </div>
              </div>

              {/* Cognitive Thinking & Rationale */}
              {trace.thinking && (
                <div className="space-y-1.5">
                  <div className="flex items-center gap-1.5 text-[11px] text-navy-500 font-semibold uppercase tracking-wider">
                    <Brain className="w-3.5 h-3.5 text-saffron" />
                    <span>Cognitive Reasoning</span>
                  </div>
                  <div className="p-2.5 bg-sky-base/30 rounded-lg border border-sky-100 text-[11px] leading-relaxed text-navy-700 font-sans">
                    {trace.thinking}
                  </div>
                </div>
              )}

              {/* Specialist Chain */}
              <div className="space-y-1.5">
                <span className="text-[11px] text-navy-400 font-semibold uppercase tracking-wider block">
                  Invoked Specialists
                </span>
                <div className="space-y-1">
                  {trace.models_used.map((model, i) => (
                    <div
                      key={i}
                      className="p-2 bg-white rounded border border-navy-200 flex items-center justify-between shadow-2xs"
                    >
                      <span className="font-medium text-navy-800">{model}</span>
                      <span className="text-[10px] text-navy-400 bg-navy-50 px-1.5 py-0.5 rounded border border-navy-100">
                        Node #{i + 1}
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Execution Telemetry */}
              <div className="space-y-1.5">
                <span className="text-[11px] text-navy-400 font-semibold uppercase tracking-wider block">
                  Telemetry
                </span>
                <div className="grid grid-cols-2 gap-2 text-[11px]">
                  <div className="p-2 bg-navy-50/70 rounded border border-navy-100">
                    <span className="text-navy-400 block text-[10px]">Latency</span>
                    <span className="font-bold text-navy-800">{trace.duration_ms.toFixed(1)} ms</span>
                  </div>
                  <div className="p-2 bg-navy-50/70 rounded border border-navy-100">
                    <span className="text-navy-400 block text-[10px]">Input Count</span>
                    <span className="font-bold text-navy-800">{trace.input_count}</span>
                  </div>
                </div>
                <div className="p-2 bg-navy-50/70 rounded border border-navy-100 text-[10px] text-navy-500 truncate">
                  <span className="text-navy-400">Timestamp:</span> {trace.timestamp}
                </div>
              </div>

              {/* Parameters JSON */}
              <div className="space-y-1.5">
                <span className="text-[11px] text-navy-400 font-semibold uppercase tracking-wider block">
                  Node Parameters
                </span>
                <div className="bg-navy-900 text-navy-100 p-3 rounded-lg overflow-x-auto text-[11px] leading-relaxed shadow-inner">
                  <pre className="whitespace-pre-wrap break-all">
                    {JSON.stringify(trace.parameters, null, 2)}
                  </pre>
                </div>
              </div>
            </>
          ) : (
            <div className="h-full flex flex-col items-center justify-center text-center text-navy-400 p-4 space-y-2">
              <Activity className="w-8 h-8 text-navy-300 animate-pulse" />
              <p className="text-xs">Awaiting query execution...</p>
              <p className="text-[10px] text-navy-300">
                Agent execution telemetry and model parameter logs will stream here.
              </p>
            </div>
          )}
        </div>
      ) : (
        <div className="flex-1 flex flex-col items-center py-6 text-navy-400 gap-4">
          <Terminal className="w-4 h-4 text-navy-400" />
          <span className="rotate-90 whitespace-nowrap text-[10px] font-mono tracking-widest uppercase">
            Trace Stream
          </span>
        </div>
      )}
    </aside>
  )
}
