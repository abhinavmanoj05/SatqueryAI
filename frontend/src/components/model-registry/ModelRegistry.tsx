import { useRef } from 'react'
import { motion, useInView } from 'framer-motion'
import { CheckCircle2, Clock, AlertCircle } from 'lucide-react'

const MODELS = [
  {
    name: 'ResNet-18',
    checkpoint: 'BIFOLD-BigEarthNetv2-0/resnet18-all-v0.2.0',
    role: 'Optical–SAR land-cover classification & sensor prior generator',
    task: 'Land Cover / Fusion Prior',
    status: 'tested' as const,
    detail: '12-channel Sentinel-1+2 input · 19 CORINE classes · Frozen weights · Macro AP ≈ 0.71',
    accent: 'border-emerald-300 bg-emerald-50',
    badge: 'text-emerald-700 bg-emerald-100',
  },
  {
    name: 'PaliGemma-3B',
    checkpoint: 'paligemma-ben-caption-v1',
    role: 'Scene captioning via QLoRA fine-tuning on BigEarthNet.txt',
    task: 'Image Captioning',
    status: 'progress' as const,
    detail: 'QLoRA attn layers · BigEarthNet.txt caption splits (100%) · T4 16 GB',
    accent: 'border-amber-300 bg-amber-50',
    badge: 'text-amber-700 bg-amber-100',
  },
  {
    name: 'InternVL2-8B',
    checkpoint: 'internvl2-vrsbench-v1',
    role: 'High-resolution visual question answering & region grounding',
    task: 'VQA + Grounding',
    status: 'pending' as const,
    detail: 'QLoRA · VRSBench 40% + RSVQA 30% + BigEarthNet 30% · 4-bit quantisation',
    accent: 'border-violet-300 bg-violet-50',
    badge: 'text-violet-700 bg-violet-100',
  },
  {
    name: 'Qwen2-VL-7B',
    checkpoint: 'qwen2vl-cdvqa-v1',
    role: 'Bi-temporal change detection & change-VQA',
    task: 'Change Detection',
    status: 'pending' as const,
    detail: 'QLoRA 4-bit · CDVQA dataset (100%) · Native multi-image input',
    accent: 'border-sky-300 bg-sky-50',
    badge: 'text-sky-700 bg-sky-100',
  },
  {
    name: 'CDVQA Baseline',
    checkpoint: '—',
    role: 'Pixel-level change mask generation for bi-temporal pairs',
    task: 'Change Mask',
    status: 'pending' as const,
    detail: 'Dual-temporal encoder · Change area statistics · Red-overlay mask output',
    accent: 'border-navy-200 bg-navy-50',
    badge: 'text-navy-700 bg-navy-100',
  },
]

const STATUS_META = {
  tested: {
    label: 'Tested',
    icon: CheckCircle2,
    cls: 'text-emerald-600',
  },
  progress: {
    label: 'In Progress',
    icon: Clock,
    cls: 'text-amber-600',
  },
  pending: {
    label: 'Pending',
    icon: AlertCircle,
    cls: 'text-navy-400',
  },
}

function ModelCard({
  model,
  index,
}: {
  model: (typeof MODELS)[number]
  index: number
}) {
  const ref = useRef<HTMLDivElement>(null)
  const inView = useInView(ref, { once: true, margin: '-50px' })
  const status = STATUS_META[model.status]
  const StatusIcon = status.icon

  return (
    <motion.article
      ref={ref}
      initial={{ opacity: 0, y: 24 }}
      animate={inView ? { opacity: 1, y: 0 } : {}}
      transition={{ duration: 0.45, delay: index * 0.08, ease: 'easeOut' }}
      className={`rounded-2xl border ${model.accent} p-6 flex flex-col gap-4 shadow-card hover:shadow-card-hover transition-shadow duration-200`}
      aria-label={`Model: ${model.name}`}
    >
      {/* Header */}
      <div className="flex items-start justify-between gap-3">
        <div>
          <h3 className="text-lg font-bold text-navy leading-tight">{model.name}</h3>
          <span className={`inline-block mt-1 px-2 py-0.5 rounded-md text-xs font-semibold ${model.badge}`}>
            {model.task}
          </span>
        </div>
        <div className={`flex items-center gap-1 text-xs font-medium ${status.cls} shrink-0`}>
          <StatusIcon className="w-3.5 h-3.5" aria-hidden="true" />
          <span>{status.label}</span>
        </div>
      </div>

      {/* Role description */}
      <p className="text-sm text-navy-700 leading-relaxed">{model.role}</p>

      {/* Technical detail */}
      <p className="text-xs text-navy-500 leading-relaxed border-t border-current border-opacity-10 pt-3">
        {model.detail}
      </p>

      {/* Checkpoint */}
      {model.checkpoint !== '—' && (
        <code className="text-xs font-mono bg-white/70 text-navy-600 px-2 py-1 rounded border border-navy-100 break-all">
          {model.checkpoint}
        </code>
      )}
    </motion.article>
  )
}

export default function ModelRegistry() {
  const titleRef = useRef<HTMLDivElement>(null)
  const titleInView = useInView(titleRef, { once: true, margin: '-60px' })

  return (
    <section id="models" className="py-24 bg-sky-base section-pad">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <motion.div
          ref={titleRef}
          initial={{ opacity: 0, y: 24 }}
          animate={titleInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.5 }}
          className="text-center mb-14"
        >
          <span className="inline-block text-xs font-semibold uppercase tracking-widest text-saffron mb-4">
            Model Registry
          </span>
          <h2 className="text-4xl md:text-5xl font-bold text-navy mb-4">
            Specialist Models
          </h2>
          <p className="text-navy-500 text-lg max-w-lg mx-auto">
            Five task-specific models — each fine-tuned for remote sensing —
            orchestrated by the LangGraph agent.
          </p>
        </motion.div>

        {/* Grid — 3 cols on xl, 2 on md, 1 on mobile */}
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-5">
          {MODELS.map((model, i) => (
            <ModelCard key={model.name} model={model} index={i} />
          ))}
        </div>

        {/* Stats footer */}
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5, delay: 0.3 }}
          className="mt-10 flex flex-wrap justify-center gap-8 text-center"
        >
          {[
            { value: '5', label: 'Specialist Models' },
            { value: '6', label: 'Task Types' },
            { value: '~15M', label: 'Training Samples' },
            { value: '4-bit', label: 'Quantisation' },
          ].map(stat => (
            <div key={stat.label}>
              <p className="text-3xl font-bold text-navy">{stat.value}</p>
              <p className="text-sm text-navy-500 mt-0.5">{stat.label}</p>
            </div>
          ))}
        </motion.div>
      </div>
    </section>
  )
}
