/** @type {import('tailwindcss').Config} */
export default {
  content: [
    './index.html',
    './src/**/*.{js,ts,jsx,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        "stonesense-ink": "#101B16",
        "stonesense-paper": "#F3F6F1",
        "stonesense-teal": "#1F6F5C",
        "stonesense-amber": "#C97A2B",
        "stonesense-indigo": "#3B3F8C",
        "stonesense-line": "#DDE3DC",
      },
      fontFamily: {
        serif: ["Fraunces", "ui-serif", "Georgia", "serif"],
        sans: ["Inter", "ui-sans-serif", "system-ui"],
      },
    },
  },
  plugins: [],
};

