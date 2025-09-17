import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        terminal: {
          bg: '#0A0E1B',
          text: '#E8E9ED',
          accent: '#00FF88',
          success: '#00FF88',
          warning: '#FF3366',
          muted: '#6B7280'
        }
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'monospace'],
      },
      spacing: {
        '8px': '8px',
        '16px': '16px',
        '24px': '24px',
        '32px': '32px',
        '48px': '48px',
        '64px': '64px',
        '80px': '80px',
      },
      animation: {
        'cursor-blink': 'cursor-blink 1s infinite',
        'typewriter': 'typewriter 2s steps(40) 1s 1 normal both',
      },
      keyframes: {
        'cursor-blink': {
          '0%, 50%': { opacity: '1' },
          '51%, 100%': { opacity: '0' },
        },
        'typewriter': {
          'from': { width: '0' },
          'to': { width: '100%' },
        },
      }
    },
  },
  plugins: [],
}
export default config