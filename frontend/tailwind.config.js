/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        base: '#07090E',
        surface: '#0D1219',
        phosphor: '#C8C0A8',
        muted: '#4A5568',
        fake: '#FF4136',
        real: '#2ECC71',
        uncertain: '#F39C12',
        gold: '#D4AF37',
      },
      fontFamily: {
        rajdhani: ['Rajdhani', 'sans-serif'],
        dmmono: ['"DM Mono"', 'monospace'],
        jbmono: ['"JetBrains Mono"', 'monospace'],
      },
      letterSpacing: {
        widest2: '0.3em',
      },
      animation: {
        scanline: 'scanline 1.8s ease-in-out 0.3s 1',
        'fade-up': 'fadeUp 0.5s ease-out both',
      },
      keyframes: {
        scanline: {
          '0%': { transform: 'translateY(-100%)' },
          '100%': { transform: 'translateY(200%)' },
        },
        fadeUp: {
          '0%': { opacity: '0', transform: 'translateY(12px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
      },
    },
  },
  plugins: [],
}

