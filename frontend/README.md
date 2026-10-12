# StoneSense-AI Frontend Subsystem

The `frontend/` directory contains the modern React single-page application built with Vite, TypeScript, and Tailwind CSS. It provides dedicated role-based consoles for clinicians (`Hospital Dashboard`) and system engineers (`Developer Console`).

---

## Technology Stack

- **Framework**: React 18.3
- **Language**: TypeScript 5.7
- **Build Tool**: Vite 5.4
- **Routing**: React Router v6
- **Styling**: Tailwind CSS 3.4
- **HTTP Client**: Axios 1.7
- **Real-Time Communication**: Native WebSocket client (`/api/v1/ws/federated`)

---

## Directory Layout

```text
frontend/src/
├── assets/                   # Static images and icons
├── components/               # Shared UI components
│   ├── layout/               # AppLayout, Sidebar
│   ├── ErrorMessage.tsx      # Clinical error displays
│   ├── ImageUpload.tsx       # CT scan drag-and-drop uploader
│   ├── LoadingSpinner.tsx    # Async request indicators
│   ├── Navbar.tsx            # Public top navigation
│   ├── PatientForm.tsx       # Biomarker input and SHAP presentation
│   ├── PredictionCard.tsx    # Risk level indicator
│   └── ProtectedRoute.tsx    # Role-based route guard
├── context/                  # React State Contexts
│   ├── authContextDef.ts     # Auth Context interface and React Context definition
│   ├── AuthContext.tsx       # AuthProvider implementation
│   ├── hospitalContextDef.ts # Hospital Context interface and React Context definition
│   └── HospitalContext.tsx   # HospitalProvider implementation
├── hooks/                    # Reusable React hooks
│   ├── useAuth.ts            # Auth state hook
│   └── useHospital.ts        # Hospital tenant selector hook
├── layouts/                  # Marketing and main layouts
├── pages/                    # Routed Page Views
│   ├── LandingPage.tsx       # Public marketing page (/)
│   ├── LoginPage.tsx         # Sign-in page with quick-fill demo roles (/login)
│   ├── HospitalDashboard.tsx # Multi-step clinical case workflow (/hospital-dashboard)
│   ├── HospitalFederatedLearning.tsx # Hospital-side FL dashboard (/hospital/federated-learning)
│   ├── HospitalPrivateDataset.tsx    # Hospital dataset upload & validation (/hospital/private-dataset)
│   ├── RiskPredictionPage.tsx        # Biomarker risk form (/risk-prediction/:patientId)
│   ├── StoneDetectionPage.tsx        # CT image classification (/stone-detection/:patientId)
│   ├── DeveloperDashboard.tsx        # 5-tab developer console (/developer-dashboard)
│   ├── HomePage.tsx          # Secondary marketing overview (/home) with validated links
│   ├── PrivacyPolicy.tsx     # Research privacy and clinical disclaimer (/privacy)
│   └── NotFoundPage.tsx      # Fallback 404 page
├── services/                 # API Client Layer
│   ├── apiClient.ts          # Axios instance with auth interceptor
│   ├── api.ts                # Core health, prediction, and concordance API calls
│   ├── authApi.ts            # Login, token refresh, profile endpoints
│   ├── developerApi.ts       # Developer console endpoints & fallbacks
│   ├── federatedApi.ts       # Federated learning endpoints & WebSocket
│   └── hospitalApi.ts        # Hospital metadata & patient history
├── types/                    # TypeScript interfaces and contracts
└── App.tsx                   # Main route configuration (with fallback route redirects)
```

---

## Development & Build Commands

```bash
# Install dependencies
npm install

# Start development server (http://localhost:5173)
npm run dev

# Run static TypeScript type check (zero emit)
npx tsc --noEmit

# Run ESLint (enforces 0 errors, 0 warnings)
npm run lint

# Compile production bundle to dist/
npm run build
```

---

## Notable Architectural & State Notes

1. **Role Protection**: Routes are wrapped with `<ProtectedRoute allowedRoles={[...]}>`. Accessing unauthorized routes redirects to `/login`.
2. **Demo Quick-Fill Accounts**: In non-production environments, the Login page provides one-click buttons to sign in as Dr. Aisha (`hospital_user`), Developer (`developer`), or Administrator (`admin`).
3. **Server-Side Concordance Engine**: The "Evidence concordance" step on `HospitalDashboard.tsx` is powered by the `/api/v1/predict/concordance` endpoint, evaluating transparent rule-based concordance across server-stored prediction IDs with hospital-level tenant isolation. Displays `CONCORDANT_NO_STONE_EVIDENCE` ("No stone evidence in either stream") with explicit disclaimers citing audited CT stone sensitivity (65.70%) and tabular sample limits ($N=79$).
4. **Normalized Explainability Labels**: All visual attention heatmaps and explanatory references are normalized to **Grad-CAM** (Selvaraju et al., 2017).
5. **Lint & Component Hygiene**: Context definitions and hooks are modularized into standalone `.ts` files, satisfying `react-refresh/only-export-components` with zero ESLint warnings across the entire codebase.
