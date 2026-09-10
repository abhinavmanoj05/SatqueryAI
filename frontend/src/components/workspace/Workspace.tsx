import { useState, useEffect } from 'react'
import Sidebar from './Sidebar'
import UploadZone from './UploadZone'
import ChatInput from './ChatInput'
import ResponsePanel from './ResponsePanel'
import AgentTracePanel from './AgentTracePanel'
import { runSatQuery, getAvailableModels, type SystemModelStatus } from '@/api/satquery'
import { Cpu, Sparkles } from 'lucide-react'
import type { UploadMode, UploadedFile, Session, ChatMessage, OrchestratorResponse } from '@/types/orchestrator'

export default function Workspace() {
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [tracePanelOpen, setTracePanelOpen] = useState(true)
  const [uploadMode, setUploadMode] = useState<UploadMode>('single')
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [preferredModel, setPreferredModel] = useState<string>('auto')
  const [systemModels, setSystemModels] = useState<SystemModelStatus | null>(null)

  useEffect(() => {
    getAvailableModels().then((data) => {
      if (data) {
        setSystemModels(data)
      }
    })
  }, [])

  // Seed sessions
  const [sessions, setSessions] = useState<Session[]>([
    {
      id: 'session-1',
      title: 'Agricultural Land Classification & LAI',
      createdAt: 'Today, 11:20 AM',
      messages: [],
    },
    {
      id: 'session-2',
      title: 'Bi-Temporal Delta & Urban Sprawl',
      createdAt: 'Yesterday, 04:15 PM',
      messages: [],
    },
  ])
  const [activeSessionId, setActiveSessionId] = useState('session-1')

  const activeSession = sessions.find((s) => s.id === activeSessionId) || sessions[0]
  const latestResponse: OrchestratorResponse | undefined =
    activeSession.messages
      .slice()
      .reverse()
      .find((m) => m.role === 'assistant' && m.response)?.response

  const handleNewSession = () => {
    const newId = `session-${Date.now()}`
    const newSession: Session = {
      id: newId,
      title: 'New Satellite Query',
      createdAt: 'Just now',
      messages: [],
    }
    setSessions((prev) => [newSession, ...prev])
    setActiveSessionId(newId)
    setUploadedFiles([])
  }

  const handleAddFiles = (newFiles: UploadedFile[]) => {
    if (uploadMode === 'single') {
      setUploadedFiles(newFiles.slice(0, 1))
    } else {
      setUploadedFiles((prev) => {
        const combined = [...prev, ...newFiles]
        return combined.slice(-2)
      })
    }
  }

  const handleRemoveFile = (index: number) => {
    setUploadedFiles((prev) => prev.filter((_, i) => i !== index))
  }

  const handleSendQuery = async (query: string) => {
    const userMsg: ChatMessage = {
      id: `msg-${Date.now()}`,
      role: 'user',
      content: query,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    }

    // Add user message to session
    setSessions((prev) =>
      prev.map((s) => {
        if (s.id === activeSessionId) {
          const updatedTitle = s.messages.length === 0 ? query.slice(0, 36) + (query.length > 36 ? '...' : '') : s.title
          return {
            ...s,
            title: updatedTitle,
            messages: [...s.messages, userMsg],
          }
        }
        return s
      })
    )

    setIsLoading(true)

    try {
      const files = uploadedFiles.map((f) => f.file)
      const orchestratorResult = await runSatQuery(query, files, undefined, preferredModel)

      const assistantMsg: ChatMessage = {
        id: `msg-${Date.now()}-res`,
        role: 'assistant',
        content: orchestratorResult.final_answer,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        response: orchestratorResult,
      }

      setSessions((prev) =>
        prev.map((s) => {
          if (s.id === activeSessionId) {
            return {
              ...s,
              messages: [...s.messages, assistantMsg],
            }
          }
          return s
        })
      )
    } catch (err) {
      console.error('Execution error:', err)
      const errorMsg: ChatMessage = {
        id: `msg-${Date.now()}-err`,
        role: 'assistant',
        content: `⚠️ Error executing request: ${err instanceof Error ? err.message : String(err)}`,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      }
      setSessions((prev) =>
        prev.map((s) => {
          if (s.id === activeSessionId) {
            return {
              ...s,
              messages: [...s.messages, errorMsg],
            }
          }
          return s
        })
      )
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <section id="workspace" className="min-h-screen bg-sky-base/40 pt-16 flex flex-col">
      {/* Workspace Subheader */}
      <div className="bg-white border-b border-navy-100 px-6 py-2.5 flex items-center justify-between flex-wrap gap-2">
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
          <h2 className="text-xs font-semibold text-navy tracking-wide uppercase">
            Unified Analysis Workspace
          </h2>
          <span className="text-navy-300">|</span>
          <span className="text-xs text-navy-500 font-medium truncate max-w-xs">
            {activeSession.title}
          </span>
        </div>

        {/* Dynamic Model Engine & Modality Selector */}
        <div className="flex items-center gap-3 text-xs text-navy-600 flex-wrap">
          <div className="flex items-center gap-1.5 bg-navy-50/80 px-2.5 py-1 rounded-md border border-navy-200">
            <Sparkles className="w-3.5 h-3.5 text-saffron" />
            <span className="text-[11px] font-semibold text-navy-700">Model Engine:</span>
            <select
              value={preferredModel}
              onChange={(e) => setPreferredModel(e.target.value)}
              className="font-mono text-[11px] bg-transparent text-navy-900 font-semibold focus:outline-none cursor-pointer"
            >
              <option value="auto">⚡ Auto (Omni-Route Dynamic)</option>
              <option value="gemini-3.8-flash">✨ Google Gemini 3.8 Flash</option>
              <option value="gemini-2.0-flash">🌐 Google Gemini 2.0 Flash</option>
              {systemModels?.ollama.online && systemModels.ollama.models.map((m) => (
                <option key={m.name} value={`ollama:${m.name}`}>
                  🦙 Ollama: {m.name} {m.parameter_size ? `[${m.parameter_size}]` : ''}
                </option>
              ))}
              <option value="vit_base">🔬 ViT-Base (BigEarthNet 12-Band S2)</option>
            </select>
          </div>

          <div className="flex items-center gap-1.5">
            <span className="hidden sm:inline text-navy-400 font-medium">Modality:</span>
            <span className="font-mono text-[11px] bg-navy-50 px-2 py-0.5 rounded border border-navy-200 uppercase font-semibold text-navy-700">
              {uploadMode}
            </span>
          </div>
        </div>
      </div>

      {/* 3-Column Product Workspace */}
      <div className="flex-1 flex overflow-hidden h-[calc(100vh-6.5rem)]">
        {/* Left Sidebar */}
        <Sidebar
          sessions={sessions}
          activeSessionId={activeSessionId}
          onSelectSession={setActiveSessionId}
          onNewSession={handleNewSession}
          isOpen={sidebarOpen}
          onToggle={() => setSidebarOpen((v) => !v)}
        />

        {/* Center Panel (Upload + Response + Input) */}
        <main className="flex-1 flex flex-col bg-white overflow-hidden min-w-0">
          {/* Collapsible Image Upload Zone */}
          <div className="p-3 border-b border-navy-100 bg-sky-base/20">
            <UploadZone
              mode={uploadMode}
              onModeChange={(m) => {
                setUploadMode(m)
                setUploadedFiles([])
              }}
              files={uploadedFiles}
              onAddFiles={handleAddFiles}
              onRemoveFile={handleRemoveFile}
            />
          </div>

          {/* Response Feed Area */}
          <ResponsePanel messages={activeSession.messages} isLoading={isLoading} />

          {/* Chat / Query Bar */}
          <ChatInput onSend={handleSendQuery} isLoading={isLoading} />
        </main>

        {/* Right Agent Trace Panel */}
        <AgentTracePanel
          trace={latestResponse?.execution_trace}
          isOpen={tracePanelOpen}
          onToggle={() => setTracePanelOpen((v) => !v)}
        />
      </div>
    </section>
  )
}
