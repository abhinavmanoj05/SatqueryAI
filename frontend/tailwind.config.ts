import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    './index.html',
    './src/**/*.{js,ts,jsx,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        navy: {
          DEFAULT: '#0B1F4B',
          50:  '#EEF1F8',
          100: '#D5DCEE',
          200: '#ABB9DD',
          300: '#8095CB',
          400: '#5672BA',
          500: '#2B4FA9',
          600: '#1A3A88',
          700: '#122A66',
          800: '#0B1F4B',
          900: '#060F25',
        },
        saffron: {
          DEFAULT: '#E85D25',
          50:  '#FEF2EC',
          100: '#FDE0CF',
          200: '#FAC19F',
          300: '#F7A26F',
          400: '#F4833F',
          500: '#E85D25',
          600: '#C44B1B',
          700: '#9A3A14',
          800: '#70290E',
          900: '#461907',
        },
        'sky-base': '#F0F4FF',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'Consolas', 'monospace'],
      },
      animation: {
        'spin-slow': 'spin 20s linear infinite',
        'fade-up': 'fadeUp 0.6s ease-out forwards',
        'pulse-slow': 'pulse 3s ease-in-out infinite',
      },
      keyframes: {
        fadeUp: {
          '0%': { opacity: '0', transform: 'translateY(24px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
      },
      backgroundImage: {
        'hero-gradient': 'linear-gradient(160deg, #EEF2FF 0%, #F8FAFF 40%, #FFFFFF 100%)',
        'card-gradient': 'linear-gradient(135deg, #FFFFFF 0%, #F8FAFF 100%)',
      },
      boxShadow: {
        'card': '0 1px 3px rgba(11,31,75,0.06), 0 4px 16px rgba(11,31,75,0.08)',
        'card-hover': '0 4px 12px rgba(11,31,75,0.10), 0 12px 32px rgba(11,31,75,0.12)',
        'input': '0 0 0 3px rgba(232,93,37,0.15)',
      },
    },
  },
  plugins: [],
}

export default config
