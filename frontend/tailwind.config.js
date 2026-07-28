/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        dark: {
          900: '#0d0d0d',
          800: '#1a1a1a',
          700: '#242424',
          600: '#2e2e2e',
          500: '#3a3a3a',
          400: '#4a4a4a',
        },
        accent: {
          DEFAULT: '#4f8ef7',
          hover: '#3a7de8',
        },
      },
    },
  },
  plugins: [],
}
