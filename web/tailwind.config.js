/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ['"Inter"', "system-ui", "-apple-system", "sans-serif"],
        mono: ['"JetBrains Mono"', '"Fira Code"', "ui-monospace", "SFMono-Regular", "monospace"],
      },
      colors: {
        obsidian: {
          950: "#07090E",
          900: "#0C1018",
          850: "#111622",
          800: "#171E2E",
          700: "#242F47",
        },
      },
    },
  },
  plugins: [],
};
