import { Plus, MessageSquare, History, ChevronLeft, ChevronRight } from 'lucide-react'
import type { Session } from '@/types/orchestrator'

interface SidebarProps {
  sessions: Session[]
  activeSessionId: string
  onSelectSession: (id: string) => void
  onNewSession: () => void
  isOpen: boolean
  onToggle: () => void
}

export default function Sidebar({
  sessions,
  activeSessionId,
  onSelectSession,
  onNewSession,
  isOpen,
  onToggle,
}: SidebarProps) {
  return (
    <aside
      className={`relative transition-all duration-300 ease-in-out border-r border-navy-100 bg-white flex flex-col z-20 ${
        isOpen ? 'w-64' : 'w-14'
      }`}
    >
      {/* Header / Collapse Toggle */}
      <div className="p-3 border-b border-navy-100 flex items-center justify-between">
        {isOpen ? (
          <div className="flex items-center gap-2 text-xs font-semibold text-navy-500 uppercase tracking-wider">
            <History className="w-4 h-4 text-navy-400" />
            <span>Analysis Sessions</span>
          </div>
        ) : (
          <div className="w-full flex justify-center">
            <History className="w-4 h-4 text-navy-400" />
          </div>
        )}
        <button
          onClick={onToggle}
          className="p-1 rounded-md text-navy-400 hover:text-navy hover:bg-navy-50 transition-colors"
          title={isOpen ? 'Collapse Sidebar' : 'Expand Sidebar'}
          aria-label={isOpen ? 'Collapse Sidebar' : 'Expand Sidebar'}
        >
          {isOpen ? <ChevronLeft className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
        </button>
      </div>

      {/* New Analysis Action */}
      <div className="p-3">
        <button
          onClick={onNewSession}
          className={`w-full flex items-center justify-center gap-2 py-2 px-3 rounded-lg border border-navy-200 hover:border-saffron hover:bg-saffron-50 text-navy hover:text-saffron transition-all duration-200 text-sm font-medium shadow-sm ${
            !isOpen ? 'px-0' : ''
          }`}
          title="New Analysis Session"
        >
          <Plus className="w-4 h-4 text-saffron shrink-0" />
          {isOpen && <span>New Analysis</span>}
        </button>
      </div>

      {/* Session History List */}
      <div className="flex-1 overflow-y-auto px-2 py-1 space-y-1">
        {sessions.map((session) => {
          const isActive = session.id === activeSessionId
          return (
            <button
              key={session.id}
              onClick={() => onSelectSession(session.id)}
              className={`w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-left transition-colors text-xs ${
                isActive
                  ? 'bg-navy-50 text-navy font-semibold border-l-2 border-saffron'
                  : 'text-navy-600 hover:bg-navy-50/60 hover:text-navy'
              }`}
              title={session.title}
            >
              <MessageSquare
                className={`w-3.5 h-3.5 shrink-0 ${isActive ? 'text-saffron' : 'text-navy-400'}`}
              />
              {isOpen && (
                <div className="flex-1 truncate">
                  <div className="truncate">{session.title}</div>
                  <div className="text-[10px] text-navy-400 font-normal">{session.createdAt}</div>
                </div>
              )}
            </button>
          )
        })}
      </div>

      {/* Bottom Info */}
      {isOpen && (
        <div className="p-3 border-t border-navy-100 bg-navy-50/50 text-[11px] text-navy-500 flex items-center justify-between">
          <span>ISRO Multimodal Agent</span>
          <span className="font-mono text-[10px] bg-white px-1.5 py-0.5 rounded border border-navy-200">v1.0</span>
        </div>
      )}
    </aside>
  )
}
