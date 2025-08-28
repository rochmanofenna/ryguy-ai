/** @type {import('tailwindcss').Config} */
export default {
  darkMode: ["class"],
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: { 
    extend: { 
      boxShadow: { 
        glow: "0 0 32px rgba(16,185,129,.35)" 
      } 
    } 
  },
  plugins: [],
}