import type { Config } from 'tailwindcss'
import typography from '@tailwindcss/typography'

export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        background: 'hsl(240 6% 8%)',
        foreground: 'hsl(0 0% 98%)',
        muted: 'hsl(240 4% 16%)',
        'muted-foreground': 'hsl(240 5% 65%)',
        border: 'hsl(240 4% 22%)',
        accent: 'hsl(240 5% 26%)',
        primary: 'hsl(217 91% 60%)',
        'primary-foreground': 'hsl(0 0% 100%)',
        danger: 'hsl(0 72% 55%)',
      },
      fontFamily: {
        sans: ['var(--app-font)', 'system-ui', 'sans-serif'],
        mono: ['var(--app-font)', 'ui-monospace', 'monospace'],
      },
    },
  },
  plugins: [typography],
} satisfies Config
