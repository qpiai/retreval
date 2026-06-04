import type { Config } from "tailwindcss";

const config: Config = {
  // Dark out of the box (set on <html> in layout.tsx); the header toggle
  // removes the class for a light theme.
  darkMode: "class",
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        bg: { 0: "#07070b", 1: "#0c0c14", 2: "#13131f" },
        edge: { 0: "#23233a", 1: "#33334d" },
        accent: "#ec4899", // pink
        accent2: "#8b5cf6", // violet
        accent3: "#22d3ee", // cyan
      },
      fontFamily: {
        sans: ["var(--font-inter)", "Inter", "system-ui", "sans-serif"],
        mono: ["var(--font-mono)", "JetBrains Mono", "ui-monospace", "monospace"],
      },
      keyframes: {
        pulseGlow: {
          "0%,100%": { opacity: "0.55" },
          "50%": { opacity: "1" },
        },
      },
      animation: {
        pulseGlow: "pulseGlow 1.6s ease-in-out infinite",
      },
    },
  },
  plugins: [],
};

export default config;
