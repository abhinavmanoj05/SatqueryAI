import { useState, useRef } from 'react'
import { UploadCloud, Image as ImageIcon, X, Layers, Clock, Orbit } from 'lucide-react'
import type { UploadMode, UploadedFile } from '@/types/orchestrator'

interface UploadZoneProps {
  mode: UploadMode
  onModeChange: (mode: UploadMode) => void
  files: UploadedFile[]
  onAddFiles: (files: UploadedFile[]) => void
  onRemoveFile: (index: number) => void
}

export default function UploadZone({
  mode,
  onModeChange,
  files,
  onAddFiles,
  onRemoveFile,
}: UploadZoneProps) {
  const [isDragging, setIsDragging] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const modeSpecs = {
    single: {
      title: 'Single Satellite Scene',
      desc: 'Analyze land cover, VQA, scene captioning, or visual grounding.',
      maxFiles: 1,
      expectedTags: ['OPTICAL'] as const,
      icon: Layers,
    },
    bitemporal: {
      title: 'Bi-temporal Pair (T0 & T1)',
      desc: 'Pair two images across different acquisition dates to detect changes.',
      maxFiles: 2,
      expectedTags: ['T0', 'T1'] as const,
      icon: Clock,
    },
    fusion: {
      title: 'Optical + SAR Fusion',
      desc: 'Combine co-registered Optical & Sentinel-1 SAR polarimetric data.',
      maxFiles: 2,
      expectedTags: ['OPTICAL', 'SAR'] as const,
      icon: Orbit,
    },
  }

  const currentSpec = modeSpecs[mode]

  const processFiles = (selectedFiles: FileList | File[]) => {
    const newItems: UploadedFile[] = []
    const availableSlots = currentSpec.maxFiles - files.length
    const countToTake = Math.min(availableSlots, selectedFiles.length)

    for (let i = 0; i < countToTake; i++) {
      const file = selectedFiles[i]
      const currentIdx = files.length + i
      let modality: 'OPTICAL' | 'SAR' | 'T0' | 'T1' = 'OPTICAL'

      if (mode === 'bitemporal') {
        modality = currentIdx === 0 ? 'T0' : 'T1'
      } else if (mode === 'fusion') {
        modality = currentIdx === 0 ? 'OPTICAL' : 'SAR'
      } else {
        modality = 'OPTICAL'
      }

      newItems.push({
        file,
        preview: URL.createObjectURL(file),
        modality,
      })
    }

    if (newItems.length > 0) {
      onAddFiles(newItems)
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      processFiles(e.dataTransfer.files)
    }
  }

  return (
    <div className="bg-white rounded-xl border border-navy-100 p-4 shadow-sm">
      {/* Mode Selector Tabs */}
      <div className="flex items-center justify-between flex-wrap gap-2 border-b border-navy-100 pb-3 mb-3">
        <div className="flex items-center gap-1.5 p-1 bg-navy-50 rounded-lg">
          {(['single', 'bitemporal', 'fusion'] as UploadMode[]).map((tabMode) => {
            const isSelected = mode === tabMode
            const TabIcon = modeSpecs[tabMode].icon
            return (
              <button
                key={tabMode}
                onClick={() => {
                  onModeChange(tabMode)
                }}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                  isSelected
                    ? 'bg-white text-navy font-semibold shadow-sm text-saffron'
                    : 'text-navy-600 hover:text-navy hover:bg-white/50'
                }`}
              >
                <TabIcon className="w-3.5 h-3.5" />
                <span className="capitalize">{tabMode === 'bitemporal' ? 'Bi-Temporal' : tabMode === 'fusion' ? 'Optical + SAR' : 'Single Image'}</span>
              </button>
            )
          })}
        </div>

        <span className="text-[11px] text-navy-400 font-mono">
          Slot: {files.length}/{currentSpec.maxFiles} images
        </span>
      </div>

      {/* Upload Dropzone */}
      <div
        onDragOver={(e) => {
          e.preventDefault()
          setIsDragging(true)
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
        onClick={() => {
          if (files.length < currentSpec.maxFiles) {
            fileInputRef.current?.click()
          }
        }}
        className={`relative border-2 border-dashed rounded-lg p-5 text-center transition-all cursor-pointer ${
          isDragging
            ? 'border-saffron bg-saffron-50/50'
            : files.length >= currentSpec.maxFiles
            ? 'border-navy-100 bg-navy-50/30 cursor-default'
            : 'border-navy-200 hover:border-navy-400 bg-white hover:bg-navy-50/30'
        }`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".tif,.tiff,.png,.jpg,.jpeg"
          multiple={currentSpec.maxFiles > 1}
          className="hidden"
          onChange={(e) => {
            if (e.target.files) {
              processFiles(e.target.files)
            }
          }}
        />

        {files.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-2 text-navy-600">
            <UploadCloud className="w-8 h-8 text-navy-400 mb-2" />
            <p className="text-sm font-medium text-navy">
              Drop {currentSpec.title.toLowerCase()} here, or <span className="text-saffron underline underline-offset-2">browse</span>
            </p>
            <p className="text-xs text-navy-400 mt-1 max-w-sm">
              Supports GeoTIFF, TIFF, PNG, or JPEG. Max {currentSpec.maxFiles} {currentSpec.maxFiles > 1 ? 'images' : 'image'}.
            </p>
          </div>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-left">
            {files.map((fileObj, idx) => (
              <div
                key={idx}
                className="flex items-center gap-3 bg-white p-2.5 rounded-lg border border-navy-100 shadow-sm relative group"
                onClick={(e) => e.stopPropagation()}
              >
                <div className="w-14 h-14 bg-navy-50 rounded border border-navy-100 overflow-hidden flex items-center justify-center shrink-0">
                  {fileObj.preview ? (
                    <img
                      src={fileObj.preview}
                      alt={fileObj.file.name}
                      className="w-full h-full object-cover"
                    />
                  ) : (
                    <ImageIcon className="w-6 h-6 text-navy-400" />
                  )}
                </div>

                <div className="flex-1 min-w-0 pr-6">
                  <div className="flex items-center gap-1.5 mb-1">
                    <span className="px-1.5 py-0.5 rounded text-[10px] font-bold tracking-wider uppercase bg-navy-100 text-navy-800">
                      {fileObj.modality}
                    </span>
                    <span className="text-[10px] text-navy-400 font-mono">
                      {(fileObj.file.size / (1024 * 1024)).toFixed(2)} MB
                    </span>
                  </div>
                  <p className="text-xs font-medium text-navy truncate" title={fileObj.file.name}>
                    {fileObj.file.name}
                  </p>
                </div>

                <button
                  onClick={(e) => {
                    e.stopPropagation()
                    onRemoveFile(idx)
                  }}
                  className="absolute top-2 right-2 p-1 rounded-md text-navy-400 hover:text-red-600 hover:bg-red-50 transition-colors"
                  title="Remove image"
                >
                  <X className="w-3.5 h-3.5" />
                </button>
              </div>
            ))}

            {files.length < currentSpec.maxFiles && (
              <div className="flex items-center justify-center border-2 border-dashed border-navy-200 rounded-lg p-4 text-navy-500 hover:border-saffron hover:text-saffron transition-all">
                <span className="text-xs font-medium">
                  + Add {mode === 'bitemporal' ? 'T1 Post-event image' : 'Sentinel-1 SAR image'}
                </span>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
