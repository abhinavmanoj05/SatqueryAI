import { useState, useEffect, useRef } from 'react'
import { Send, Mic, MicOff, Sparkles, Loader2, Paperclip, X, Layers } from 'lucide-react'
import { useVoiceInput } from '@/hooks/useVoiceInput'

export interface AttachedFileItem {
  id: string
  file: File
  preview: string
  isTiff: boolean
  sizeStr: string
}

interface ChatInputProps {
  onSend: (query: string, attachedFiles?: File[]) => void
  isLoading: boolean
  placeholder?: string
}

export default function ChatInput({
  onSend,
  isLoading,
  placeholder = 'Ask a question about the satellite scene or request a specific analysis...',
}: ChatInputProps) {
  const [query, setQuery] = useState('')
  const [attachedFiles, setAttachedFiles] = useState<AttachedFileItem[]>([])
  const [isDragOver, setIsDragOver] = useState(false)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const { isListening, transcript, isSupported, start, stop, clear } = useVoiceInput()

  useEffect(() => {
    if (transcript) {
      setQuery(transcript)
    }
  }, [transcript])

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFiles(e.target.files)
      e.target.value = ''
    }
  }

  const handleFiles = (files: FileList | File[]) => {
    const arr = Array.from(files)
    if (arr.length === 0) return
    const newItems: AttachedFileItem[] = arr.map((f) => {
      const isTiff = /\.(tif|tiff|geotiff)$/i.test(f.name)
      return {
        id: `${f.name}-${Date.now()}-${Math.random().toString(36).substring(2, 7)}`,
        file: f,
        preview: isTiff ? '' : URL.createObjectURL(f),
        isTiff,
        sizeStr: `${(f.size / (1024 * 1024)).toFixed(2)} MB`,
      }
    })
    setAttachedFiles((prev) => [...prev, ...newItems])
  }

  const removeAttachedFile = (id: string) => {
    setAttachedFiles((prev) => {
      const item = prev.find((p) => p.id === id)
      if (item && item.preview) {
        URL.revokeObjectURL(item.preview)
      }
      return prev.filter((p) => p.id !== id)
    })
  }

  const handlePaste = (e: React.ClipboardEvent) => {
    if (e.clipboardData && e.clipboardData.files.length > 0) {
      e.preventDefault()
      handleFiles(e.clipboardData.files)
    }
  }

  const handleSubmit = () => {
    if (isLoading) return
    const trimmed = query.trim()
    if (!trimmed && attachedFiles.length === 0) return

    const finalQuery =
      trimmed ||
      (attachedFiles.length > 1
        ? 'Analyze and fuse these satellite images, classifying land cover and detecting variations.'
        : 'Analyze this satellite scene and identify key land-cover categories, spatial structures, and features.')

    const filesToSend = attachedFiles.map((a) => a.file)
    onSend(finalQuery, filesToSend.length > 0 ? filesToSend : undefined)
    setQuery('')
    setAttachedFiles([])
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

      {/* Query Bar with File Attachments Preview & Drag/Drop */}
      <div
        onDragOver={(e) => {
          e.preventDefault()
          setIsDragOver(true)
        }}
        onDragLeave={() => setIsDragOver(false)}
        onDrop={(e) => {
          e.preventDefault()
          setIsDragOver(false)
          if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleFiles(e.dataTransfer.files)
          }
        }}
        className={`relative flex flex-col bg-navy-50/70 focus-within:bg-white border rounded-xl p-2.5 transition-all shadow-sm ${
          isDragOver
            ? 'border-saffron bg-saffron-50/30'
            : 'border-navy-200 focus-within:border-saffron'
        }`}
      >
        {/* Attached Files Thumbnail Preview Strip */}
        {attachedFiles.length > 0 && (
          <div className="flex items-center gap-2 overflow-x-auto pb-2 mb-2 border-b border-navy-100/80">
            {attachedFiles.map((att) => (
              <div
                key={att.id}
                className="flex items-center gap-2 bg-white px-2 py-1.5 rounded-lg border border-navy-200 shadow-sm shrink-0 group relative"
              >
                {att.preview && !att.isTiff ? (
                  <img
                    src={att.preview}
                    alt={att.file.name}
                    className="w-10 h-10 object-cover rounded bg-navy-900 border border-navy-100"
                  />
                ) : (
                  <div className="w-10 h-10 rounded bg-navy-900 flex flex-col items-center justify-center text-saffron">
                    <Layers className="w-4 h-4" />
                    <span className="text-[7px] font-mono text-sky-200 uppercase font-bold">TIFF</span>
                  </div>
                )}
                <div className="text-left max-w-[130px] min-w-[70px]">
                  <p className="text-xs font-semibold text-navy-900 truncate" title={att.file.name}>
                    {att.file.name}
                  </p>
                  <span className="text-[10px] text-navy-400 font-mono block">
                    {att.sizeStr}
                  </span>
                </div>
                <button
                  type="button"
                  onClick={() => removeAttachedFile(att.id)}
                  className="w-5 h-5 rounded-full bg-navy-100 hover:bg-red-500 hover:text-white flex items-center justify-center text-navy-500 transition-colors ml-1"
                  title="Remove attachment"
                  aria-label="Remove attachment"
                >
                  <X className="w-3 h-3" />
                </button>
              </div>
            ))}
            <span className="text-[10px] text-navy-400 font-mono pl-1 shrink-0">
              {attachedFiles.length} file{attachedFiles.length > 1 ? 's' : ''} attached
            </span>
          </div>
        )}

        {/* Input Bar Row */}
        <div className="flex items-end gap-2">
          {/* Hidden File Input */}
          <input
            ref={fileInputRef}
            type="file"
            accept=".tif,.tiff,.png,.jpg,.jpeg,.geotiff,.TIF,.TIFF,.PNG,.JPG,.JPEG,image/png,image/jpeg,image/tiff"
            multiple
            className="hidden"
            onChange={handleFileSelect}
          />

          <textarea
            ref={textareaRef}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            onPaste={handlePaste}
            placeholder={
              attachedFiles.length > 0
                ? 'Ask a question about the attached image(s)...'
                : placeholder
            }
            rows={1}
            className="flex-1 bg-transparent border-0 resize-none focus:outline-none text-sm text-navy placeholder:text-navy-400 max-h-32 min-h-[2.5rem] py-1"
          />

          <div className="flex items-center gap-1 shrink-0 pb-0.5">
            {/* Attachment Button */}
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              disabled={isLoading}
              className="p-2 rounded-lg text-navy-500 hover:text-navy hover:bg-navy-100 disabled:opacity-40 transition-all relative"
              title="Attach satellite imagery (.png, .jpg, .tif)"
              aria-label="Attach satellite imagery"
            >
              <Paperclip className="w-4 h-4" />
              {attachedFiles.length > 0 && (
                <span className="absolute -top-1 -right-1 w-4 h-4 rounded-full bg-saffron text-white text-[9px] font-bold flex items-center justify-center shadow-xs">
                  {attachedFiles.length}
                </span>
              )}
            </button>

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
              disabled={(!query.trim() && attachedFiles.length === 0) || isLoading}
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
      </div>

      <div className="flex justify-between items-center text-[10px] text-navy-400 mt-1 px-1">
        <span>
          Press <kbd className="font-mono bg-navy-50 px-1 py-0.5 rounded border border-navy-200">Enter</kbd> to analyze, <kbd className="font-mono bg-navy-50 px-1 py-0.5 rounded border border-navy-200">Shift+Enter</kbd> for newline, or paste/drag images
        </span>
        <span>LangGraph Orchestrator Active</span>
      </div>
    </div>
  )
}
