# Frontend integration — Developer + Hospital dashboards

## 1. Copy files into your repo
```
frontend/src/types/dashboard.ts
frontend/src/context/HospitalContext.tsx
frontend/src/services/hospitalApi.ts
frontend/src/services/developerApi.ts
frontend/src/components/layout/Sidebar.tsx
frontend/src/components/layout/AppLayout.tsx
frontend/src/pages/HospitalDashboard.tsx
frontend/src/pages/DeveloperDashboard.tsx
```

## 2. Extend Tailwind theme
In `tailwind.config.js`, add the StoneSense palette + serif display font:

```js
module.exports = {
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
};
```

Note: Tailwind can't resolve template-literal class names like
`` `bg-${accent}/15` `` at build time (JIT needs static strings). In
`Sidebar.tsx`, either add `stonesense-teal` and `stonesense-indigo` to
your `safelist` in `tailwind.config.js`, or replace that one line with an
explicit ternary — whichever you prefer, flagged inline in the file.

## 3. Load fonts
In `frontend/index.html` `<head>`:
```html
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Fraunces:wght@400;500;600&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
```

## 4. Wrap the app with HospitalProvider and add routes
In `main.tsx` (or wherever `<App />` mounts):
```tsx
import { HospitalProvider } from "./context/HospitalContext";

<HospitalProvider>
  <App />
</HospitalProvider>
```

In your router config, add:
```tsx
<Route path="/hospital-dashboard" element={<HospitalDashboard />} />
<Route path="/developer-dashboard" element={<DeveloperDashboard />} />
```
Your existing `/`, `/risk-prediction`, `/stone-detection` pages stay as-is —
`HospitalDashboard` links to them via the sidebar rather than replacing them.
Point your login/landing redirect at `/hospital-dashboard` by default.

## 5. Env
Point Axios at your backend if it's not proxied already — either add a Vite
proxy for `/api` in `vite.config.ts`, or set `baseURL` in the two new service
files to `http://127.0.0.1:8000/api/v1/...`.

## 6. Visual reference
See `../preview/dashboards_preview.html` — a static, styled preview of both
dashboards using the same tokens above, for a quick visual gut-check before
wiring real data.
