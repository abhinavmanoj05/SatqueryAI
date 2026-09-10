import { Satellite, ExternalLink } from 'lucide-react'

export default function Footer() {
  return (
    <footer className="border-t border-navy-100 bg-white">
      <div className="max-w-7xl mx-auto px-6 md:px-12 py-8 flex flex-col md:flex-row items-center justify-between gap-4">
        {/* Brand + attribution */}
        <div className="flex items-center gap-2 text-sm text-navy-600">
          <Satellite className="w-4 h-4 text-saffron shrink-0" aria-hidden="true" />
          <span>
            <span className="font-semibold text-navy">SatQuery AI</span>
            {' '}— Built for{' '}
            <abbr title="Smart India Hackathon">SIH</abbr>
            <span className="font-mono text-xs">26167</span>,{' '}
            <span className="font-medium">ISRO</span>{' '}
            <span className="text-navy-400">·</span>{' '}
            Smart India Hackathon 2026
          </span>
        </div>

        {/* Links */}
        <nav aria-label="Footer navigation">
          <ul className="flex items-center gap-4 text-xs text-navy-500" role="list">
            <li>
              <a
                href="https://www.isro.gov.in"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-1 hover:text-navy transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-saffron rounded"
              >
                ISRO.gov.in
                <ExternalLink className="w-3 h-3" aria-hidden="true" />
              </a>
            </li>
            <li aria-hidden="true" className="text-navy-200">·</li>
            <li>
              <a
                href="https://www.sih.gov.in"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-1 hover:text-navy transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-saffron rounded"
              >
                SIH 2026
                <ExternalLink className="w-3 h-3" aria-hidden="true" />
              </a>
            </li>
          </ul>
        </nav>

        {/* Copyright */}
        <p className="text-xs text-navy-400">
          © 2026 SatQuery AI Team. Problem SIH26167.
        </p>
      </div>
    </footer>
  )
}
