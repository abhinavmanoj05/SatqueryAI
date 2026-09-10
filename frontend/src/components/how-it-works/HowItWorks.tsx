import { useRef } from 'react'
import { motion, useInView } from 'framer-motion'
import { Upload, Brain, GitFork, Cpu, BarChart3 } from 'lucide-react'

const STEPS = [
  {
    number: '01',
    icon: Upload,
    title: 'Upload Your Imagery',
    description:
      'Upload single optical or SAR images, bi-temporal pairs for change analysis, or fused optical+SAR datasets. Supports GeoTIFF, TIFF, PNG, and JPEG.',
    node: 'Input Gateway',
    color: 'text-sky-600',
    bg: 'bg-sky-50',
    border: 'border-sky-200',
  },
  {
    number: '02',
    icon: Brain,
    title: 'Query Understanding',
    description:
      'A lightweight NLP parser (query_parser node) converts your natural-language question into a structured intent — identifying the task type, required modalities, and whether spatial output is needed.',
    node: 'query_parser',
    color: 'text-violet-600',
    bg: 'bg-violet-50',
    border: 'border-violet-200',
  },
  {
    number: '03',
    icon: GitFork,
    title: 'Validation & Routing',
    description:
      'The validation_gate checks image count, format, and co-registration compatibility. The task_router then selects the right specialist pipeline: VQA, captioning, grounding, change detection, or fusion.',
    node: 'validation_gate → task_router',
    color: 'text-amber-600',
    bg: 'bg-amber-50',
    border: 'border-amber-200',
  },
  {
    number: '04',
    icon: Cpu,
    title: 'Specialist Model Execution',
    description:
      'The chosen specialist node runs inference — PaliGemma-3B for captioning, InternVL2-8B for VQA and grounding, Qwen2-VL-7B for temporal change, or ResNet-18+InternVL2 for optical–SAR fusion.',
    node: 'Specialist Node',
    color: 'text-emerald-600',
    bg: 'bg-emerald-50',
    border: 'border-emerald-200',
  },
  {
    number: '05',
    icon: BarChart3,
    title: 'Evidence-Grounded Output',
    description:
      'The output_combinator merges model outputs into a final answer with visual evidence (bounding boxes, change masks, probability bars) and a full auditable execution trace.',
    node: 'output_combinator',
    color: 'text-saffron',
    bg: 'bg-saffron-50',
    border: 'border-saffron-200',
  },
]

function Step({
  step,
  index,
}: {
  step: (typeof STEPS)[number]
  index: number
}) {
  const ref = useRef<HTMLDivElement>(null)
  const inView = useInView(ref, { once: true, margin: '-80px' })
  const Icon = step.icon
  const isEven = index % 2 === 0

  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: 32 }}
      animate={inView ? { opacity: 1, y: 0 } : {}}
      transition={{ duration: 0.55, delay: 0.1 * index, ease: 'easeOut' }}
      className={`flex flex-col md:flex-row items-center gap-8 md:gap-16 ${
        !isEven ? 'md:flex-row-reverse' : ''
      }`}
    >
      {/* Diagram card */}
      <div className="flex-1 flex justify-center">
        <div
          className={`w-full max-w-sm rounded-2xl border ${step.border} ${step.bg} p-8 flex flex-col items-center text-center shadow-sm`}
        >
          <div
            className={`w-14 h-14 rounded-2xl ${step.bg} border ${step.border} flex items-center justify-center mb-4`}
          >
            <Icon className={`w-7 h-7 ${step.color}`} aria-hidden="true" />
          </div>
          <span
            className={`text-xs font-mono font-semibold tracking-widest uppercase ${step.color} mb-2`}
          >
            {step.node}
          </span>
          <span className="text-3xl font-bold text-navy-100 select-none">
            {step.number}
          </span>
        </div>
      </div>

      {/* Text */}
      <div className="flex-1">
        <p className={`text-sm font-semibold uppercase tracking-widest ${step.color} mb-2`}>
          Step {step.number}
        </p>
        <h3 className="text-2xl font-bold text-navy mb-3">{step.title}</h3>
        <p className="text-navy-600 leading-relaxed">{step.description}</p>
      </div>
    </motion.div>
  )
}

export default function HowItWorks() {
  const titleRef = useRef<HTMLDivElement>(null)
  const titleInView = useInView(titleRef, { once: true, margin: '-60px' })

  return (
    <section id="how-it-works" className="py-24 bg-white section-pad">
      <div className="max-w-4xl mx-auto">
        {/* Section header */}
        <motion.div
          ref={titleRef}
          initial={{ opacity: 0, y: 24 }}
          animate={titleInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.5 }}
          className="text-center mb-20"
        >
          <span className="inline-block text-xs font-semibold uppercase tracking-widest text-saffron mb-4">
            The Pipeline
          </span>
          <h2 className="text-4xl md:text-5xl font-bold text-navy mb-4">
            How It Works
          </h2>
          <p className="text-navy-500 text-lg max-w-xl mx-auto">
            Every query flows through a five-node LangGraph orchestrator —
            no black box, full execution trace.
          </p>
        </motion.div>

        {/* Steps */}
        <div className="space-y-20">
          {STEPS.map((step, i) => (
            <Step key={step.number} step={step} index={i} />
          ))}
        </div>

        {/* Pipeline connector summary */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: '-60px' }}
          transition={{ duration: 0.5 }}
          className="mt-20 p-6 rounded-2xl bg-navy-800 text-center"
        >
          <p className="text-sm font-mono text-navy-300 tracking-wide">
            <span className="text-sky-400">query_parser</span>
            <span className="text-navy-500 mx-2">→</span>
            <span className="text-violet-400">validation_gate</span>
            <span className="text-navy-500 mx-2">→</span>
            <span className="text-amber-400">task_router</span>
            <span className="text-navy-500 mx-2">→</span>
            <span className="text-emerald-400">specialist_node</span>
            <span className="text-navy-500 mx-2">→</span>
            <span className="text-saffron">output_combinator</span>
          </p>
          <p className="text-xs text-navy-400 mt-2">LangGraph StateGraph — auditable execution trace on every response</p>
        </motion.div>
      </div>
    </section>
  )
}
