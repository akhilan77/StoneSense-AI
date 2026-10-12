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
│   ├── AuthContext.tsx       # JWT authentication & user roles
│   └── HospitalContext.tsx   # Hospital tenant selector
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
│   ├── HomePage.tsx          # Secondary marketing overview (/home)
│   ├── PrivacyPolicy.tsx     # Research privacy and clinical disclaimer (/privacy)
│   └── NotFoundPage.tsx      # Fallback 404 page
├── services/                 # API Client Layer
│   ├── apiClient.ts          # Axios instance with auth interceptor
│   ├── api.ts                # Core health and prediction API calls
│   ├── authApi.ts            # Login, token refresh, profile endpoints
│   ├── developerApi.ts       # Developer console endpoints & fallbacks
│   ├── federatedApi.ts       # Federated learning endpoints & WebSocket
│   └── hospitalApi.ts        # Hospital metadata & patient history
├── types/                    # TypeScript interfaces and contracts
└── App.tsx                   # Main route configuration
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

# Run ESLint
npm run lint

# Compile production bundle to dist/
npm run build
```

---

## Notable Architectural & State Notes

1. **Role Protection**: Routes are wrapped with `<ProtectedRoute allowedRoles={[...]}>`. Accessing unauthorized routes redirects to `/login`.
2. **Demo Quick-Fill Accounts**: In non-production environments, the Login page provides one-click buttons to sign in as Dr. Aisha (`hospital_user`), Developer (`developer`), or Administrator (`admin`).
3. **Hardcoded Concordance Warning**: The "Trustworthy assessment" step on `HospitalDashboard.tsx` currently displays static placeholder text ("Moderate risk 71%", "Stone detected 94.2%"). Dynamic rule-based agreement calculation will be wired in Phase 4.
