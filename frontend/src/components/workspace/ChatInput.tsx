import { useState, useEffect, useRef } from 'react'
import { Send, Mic, MicOff, Sparkles, Loader2 } from 'lucide-react'
import { useVoiceInput } from '@/hooks/useVoiceInput'

interface ChatInputProps {
  onSend: (query: string) => void
  isLoading: boolean
  placeholder?: string
}

export default function ChatInput({
  onSend,
  isLoading,
  placeholder = 'Ask a question about the satellite scene or request a specific analysis...',
}: ChatInputProps) {
  const [query, setQuery] = useState('')
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const { isListening, transcript, isSupported, start, stop, clear } = useVoiceInput()

  useEffect(() => {
    if (transcript) {
      setQuery(transcript)
    }
  }, [transcript])

  const handleSubmit = () => {
    if (!query.trim() || isLoading) return
    onSend(query.trim())
    setQuery('')
    clear()
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  const toggleVoice = () => {
    if (isListening) {
      stop()
    } else {
      clear()
      start()
    }
  }

  const samplePrompts = [
    'Describe the land-cover and major objects visible in this image.',
    'Highlight the water body and parcel boundaries.',
    'What changed between these two acquisition dates?',
    'Perform optical and SAR fusion to identify built-up areas.',
  ]

  return (
    <div className="bg-white border-t border-navy-100 p-4">
      {/* Sample Quick Queries */}
      <div className="flex items-center gap-1.5 overflow-x-auto pb-2 mb-2 no-scrollbar">
        <span className="flex items-center gap-1 text-[11px] text-navy-400 font-medium shrink-0 mr-1">
          <Sparkles className="w-3 h-3 text-saffron" />
          Quick queries:
        </span>
        {samplePrompts.map((prompt, i) => (
          <button
            key={i}
            onClick={() => setQuery(prompt)}
            className="text-[11px] bg-navy-50 hover:bg-navy-100 text-navy-700 px-2.5 py-1 rounded-full whitespace-nowrap transition-colors border border-navy-100"
          >
            {prompt}
          </button>
        ))}
      </div>

      {/* Query Bar */}
      <div className="relative flex items-end gap-2 bg-navy-50/70 focus-within:bg-white border border-navy-200 focus-within:border-saffron rounded-xl p-2.5 transition-all shadow-sm">
        <textarea
          ref={textareaRef}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          rows={1}
          className="flex-1 bg-transparent border-0 resize-none focus:outline-none text-sm text-navy placeholder:text-navy-400 max-h-32 min-h-[2.5rem] py-1"
        />

        <div className="flex items-center gap-1 shrink-0">
          {/* Voice Input Button */}
          {isSupported && (
            <button
              type="button"
              onClick={toggleVoice}
              className={`p-2 rounded-lg transition-all ${
                isListening
                  ? 'bg-red-500 text-white animate-pulse'
                  : 'text-navy-500 hover:text-navy hover:bg-navy-100'
              }`}
              title={isListening ? 'Listening... click to stop' : 'Voice Assistant input'}
              aria-label={isListening ? 'Stop listening' : 'Start voice input'}
            >
              {isListening ? <MicOff className="w-4 h-4" /> : <Mic className="w-4 h-4" />}
            </button>
          )}

          {/* Send Button */}
          <button
            type="button"
            onClick={handleSubmit}
            disabled={!query.trim() || isLoading}
            className="p-2 rounded-lg bg-saffron hover:bg-saffron-600 disabled:opacity-40 disabled:hover:bg-saffron text-white shadow-sm transition-colors"
            title="Execute query"
            aria-label="Send query"
          >
            {isLoading ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Send className="w-4 h-4" />
            )}
          </button>
        </div>
      </div>
      <div className="flex justify-between items-center text-[10px] text-navy-400 mt-1 px-1">
        <span>Press <kbd className="font-mono bg-navy-50 px-1 py-0.5 rounded border border-navy-200">Enter</kbd> to analyze, <kbd className="font-mono bg-navy-50 px-1 py-0.5 rounded border border-navy-200">Shift+Enter</kbd> for newline</span>
        <span>LangGraph Orchestrator Active</span>
      </div>
    </div>
  )
}
