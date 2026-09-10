import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Satellite, Menu, X } from 'lucide-react'

const NAV_LINKS = [
  { label: 'How It Works', href: '#how-it-works' },
  { label: 'Models', href: '#models' },
  { label: 'Workspace', href: '#workspace' },
]

export default function Navbar() {
  const [scrolled, setScrolled] = useState(false)
  const [menuOpen, setMenuOpen] = useState(false)

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 20)
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  const handleNav = (href: string) => {
    setMenuOpen(false)
    const el = document.querySelector(href)
    el?.scrollIntoView({ behavior: 'smooth' })
  }

  return (
    <header
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
        scrolled
          ? 'bg-white/90 backdrop-blur-md shadow-sm border-b border-navy-100'
          : 'bg-transparent'
      }`}
    >
      <nav className="max-w-7xl mx-auto px-6 md:px-12 h-14 flex items-center justify-between">
        {/* Logo */}
        <button
          onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
          className="flex items-center gap-2 group focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-saffron rounded-md"
          aria-label="SatQuery AI — scroll to top"
        >
          <Satellite
            className="w-5 h-5 text-saffron group-hover:rotate-12 transition-transform duration-300"
            aria-hidden="true"
          />
          <span className="font-semibold text-navy tracking-tight">
            SatQuery<span className="text-saffron">AI</span>
          </span>
        </button>

        {/* Desktop nav */}
        <ul className="hidden md:flex items-center gap-1" role="list">
          {NAV_LINKS.map(link => (
            <li key={link.href}>
              <button
                onClick={() => handleNav(link.href)}
                className="px-3 py-1.5 text-sm font-medium text-navy-700 hover:text-navy rounded-md hover:bg-navy-50 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-saffron"
              >
                {link.label}
              </button>
            </li>
          ))}
          <li>
            <button
              onClick={() => handleNav('#workspace')}
              className="ml-2 px-4 py-1.5 text-sm font-semibold text-white bg-saffron hover:bg-saffron-600 rounded-lg transition-colors shadow-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-saffron ring-offset-2"
            >
              Launch Workspace
            </button>
          </li>
        </ul>

        {/* Mobile menu toggle */}
        <button
          className="md:hidden p-1.5 rounded-md text-navy hover:bg-navy-50 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-saffron"
          onClick={() => setMenuOpen(v => !v)}
          aria-label={menuOpen ? 'Close menu' : 'Open menu'}
          aria-expanded={menuOpen}
        >
          {menuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
        </button>
      </nav>

      {/* Mobile dropdown */}
      <AnimatePresence>
        {menuOpen && (
          <motion.div
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.18 }}
            className="md:hidden bg-white border-b border-navy-100 shadow-sm"
          >
            <ul className="px-6 py-3 space-y-1" role="list">
              {NAV_LINKS.map(link => (
                <li key={link.href}>
                  <button
                    onClick={() => handleNav(link.href)}
                    className="block w-full text-left px-3 py-2 text-sm font-medium text-navy-700 hover:text-navy rounded-md hover:bg-navy-50 transition-colors"
                  >
                    {link.label}
                  </button>
                </li>
              ))}
              <li className="pt-1 pb-2">
                <button
                  onClick={() => handleNav('#workspace')}
                  className="block w-full text-center px-4 py-2 text-sm font-semibold text-white bg-saffron rounded-lg hover:bg-saffron-600 transition-colors"
                >
                  Launch Workspace
                </button>
              </li>
            </ul>
          </motion.div>
        )}
      </AnimatePresence>
    </header>
  )
}
