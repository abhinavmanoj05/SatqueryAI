import { useState, useCallback } from 'react'
import { motion } from 'framer-motion'
import { ChevronDown } from 'lucide-react'
import BASModel from './BASModel'

function scrollToWorkspace() {
  document.getElementById('workspace')?.scrollIntoView({ behavior: 'smooth' })
}

export default function HeroSection() {
  const [mouse, setMouse] = useState({ x: 0, y: 0 })

  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLElement>) => {
    const rect = e.currentTarget.getBoundingClientRect()
    const x = ((e.clientX - rect.left) / rect.width - 0.5) * 2  // [-1, 1]
    const y = ((e.clientY - rect.top) / rect.height - 0.5) * 2   // [-1, 1]
    setMouse({ x, y })
  }, [])

  return (
    <section
      className="relative min-h-screen flex flex-col items-center justify-center overflow-hidden"
      style={{
        background: 'linear-gradient(160deg, #EEF2FF 0%, #F5F8FF 45%, #FFFFFF 100%)',
      }}
      onMouseMove={handleMouseMove}
      aria-label="SatQuery AI hero"
    >
      {/* Subtle grid overlay */}
      <div
        className="absolute inset-0 pointer-events-none opacity-30"
        style={{
          backgroundImage: `
            linear-gradient(rgba(11,31,75,0.04) 1px, transparent 1px),
            linear-gradient(90deg, rgba(11,31,75,0.04) 1px, transparent 1px)
          `,
          backgroundSize: '60px 60px',
        }}
        aria-hidden="true"
      />

      {/* 3D BAS Model — fills the section, pointer-events off so text is clickable */}
      <div className="absolute inset-0 z-0" aria-hidden="true">
        <BASModel mouseX={mouse.x} mouseY={mouse.y} />
      </div>

      {/* Content */}
      <div className="relative z-10 max-w-4xl mx-auto px-6 text-center pt-20">
        {/* SIH badge */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
        >
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold tracking-wide bg-saffron-50 text-saffron border border-saffron-200 mb-8">
            <span className="inline-block w-1.5 h-1.5 rounded-full bg-saffron animate-pulse" />
            ISRO Smart India Hackathon 2026 · SIH26167
          </span>
        </motion.div>

        {/* Headline */}
        <motion.h1
          className="text-5xl sm:text-6xl lg:text-7xl font-bold text-navy leading-[1.08] tracking-tight text-balance mb-6"
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
        >
          Ask anything about{' '}
          <span className="text-saffron">satellite imagery.</span>
        </motion.h1>

        {/* Subheadline */}
        <motion.p
          className="text-lg sm:text-xl text-navy-600 max-w-2xl mx-auto leading-relaxed mb-10 text-balance"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.35 }}
        >
          An agentic vision-language assistant for multimodal remote sensing analysis.
          Upload optical or SAR imagery, type a natural-language query, and get
          evidence-grounded answers from a LangGraph specialist pipeline.
        </motion.p>

        {/* CTAs */}
        <motion.div
          className="flex flex-col sm:flex-row items-center justify-center gap-3"
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.5 }}
        >
          <button
            onClick={scrollToWorkspace}
            className="px-8 py-3.5 rounded-xl font-semibold text-white bg-saffron hover:bg-saffron-600 shadow-md hover:shadow-lg transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-saffron ring-offset-2 ring-offset-white text-base"
          >
            Launch Workspace
          </button>
          <button
            onClick={() => document.getElementById('how-it-works')?.scrollIntoView({ behavior: 'smooth' })}
            className="px-8 py-3.5 rounded-xl font-medium text-navy-700 bg-white/80 hover:bg-white border border-navy-200 hover:border-navy-300 shadow-sm hover:shadow transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-navy-300 text-base"
          >
            How It Works
          </button>
        </motion.div>

        {/* Capability pills */}
        <motion.div
          className="flex flex-wrap items-center justify-center gap-2 mt-10"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.6, delay: 0.7 }}
        >
          {[
            'Visual QA',
            'Scene Captioning',
            'Visual Grounding',
            'Change Detection',
            'Optical–SAR Fusion',
          ].map(cap => (
            <span
              key={cap}
              className="px-3 py-1 rounded-full text-xs font-medium bg-white/70 text-navy-700 border border-navy-200 backdrop-blur-sm"
            >
              {cap}
            </span>
          ))}
        </motion.div>
      </div>

      {/* Scroll indicator */}
      <motion.button
        onClick={scrollToWorkspace}
        aria-label="Scroll down"
        className="absolute bottom-8 left-1/2 -translate-x-1/2 z-10 text-navy-400 hover:text-navy transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-saffron rounded-full p-1"
        animate={{ y: [0, 6, 0] }}
        transition={{ duration: 2, repeat: Infinity, ease: 'easeInOut' }}
      >
        <ChevronDown className="w-6 h-6" />
      </motion.button>
    </section>
  )
}
