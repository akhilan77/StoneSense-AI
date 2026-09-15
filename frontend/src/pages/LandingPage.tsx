import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useHospital } from "../context/HospitalContext";

export function LandingPage() {
  const navigate = useNavigate();
  const { hospitals, hospitalId, setHospitalId } = useHospital();
  
  // Login flow state
  const [loginMode, setLoginMode] = useState<"select" | "hospital" | "developer">("select");

  const enterHospital = () => {
    if (hospitalId == null && hospitals.length > 0) {
      setHospitalId(hospitals[0].id);
    }
    navigate("/hospital-dashboard");
  };

  const enterDeveloper = () => {
    navigate("/developer-dashboard");
  };

  return (
    <div className="min-h-screen bg-stonesense-paper text-stonesense-ink font-sans selection:bg-stonesense-teal selection:text-white">
      {/* 1. Header/Nav */}
      <header className="border-b border-stonesense-ink/10 bg-stonesense-paper sticky top-0 z-10">
        <div className="h-1 bg-stonesense-ink w-full"></div>
        <div className="max-w-6xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="font-serif text-2xl font-medium tracking-tight">StoneSense-AI</div>
          <nav className="hidden md:flex items-center gap-8 text-sm font-medium text-stonesense-ink/70">
            <a href="#features" className="hover:text-stonesense-ink transition-colors">Features</a>
            <a href="#about" className="hover:text-stonesense-ink transition-colors">About</a>
          </nav>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-6 py-16 md:py-24 space-y-24">
        {/* 2. Hero Section */}
        <section className="text-center max-w-3xl mx-auto space-y-6">
          <h1 className="font-serif text-5xl md:text-6xl font-medium leading-tight text-stonesense-ink">
            Trustworthy AI for kidney-stone assessment
          </h1>
          <p className="text-lg md:text-xl text-stonesense-ink/70 leading-relaxed">
            An explainable decision-support platform combining urine biomarker analysis and CT scan imaging.
          </p>
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-white border border-stonesense-line text-sm text-stonesense-ink/60 shadow-sm">
            <span className="w-2 h-2 rounded-full bg-stonesense-amber"></span>
            Evidence-based decision support, not a clinical diagnosis
          </div>
        </section>

        {/* Problem & Audience */}
        <section className="grid md:grid-cols-2 gap-12 items-start max-w-5xl mx-auto border-t border-stonesense-line pt-16">
          <div className="space-y-4">
            <h2 className="font-serif text-3xl text-stonesense-ink">The Challenge</h2>
            <p className="text-stonesense-ink/70 leading-relaxed">
              Assessing kidney stones often requires analyzing complex CT scans alongside patient biomarkers. Traditional AI models act as "black boxes," making it difficult for clinicians to trust their recommendations. There is a critical need for transparent, explainable decision support that clearly shows how it arrives at its risk scores and imaging classifications.
            </p>
          </div>
          <div className="space-y-4">
            <h2 className="font-serif text-3xl text-stonesense-ink">Who It's For</h2>
            <ul className="space-y-3 text-stonesense-ink/70">
              <li className="flex items-start gap-2">
                <svg className="w-5 h-5 text-stonesense-teal mt-0.5 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" /></svg>
                <span><strong>Hospitals & Clinicians:</strong> Submit cases, review AI evidence backed by SHAP and Grad-CAM, and make informed decisions without compromising patient data.</span>
              </li>
              <li className="flex items-start gap-2">
                <svg className="w-5 h-5 text-stonesense-teal mt-0.5 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" /></svg>
                <span><strong>AI Developers:</strong> Monitor aggregate system health, track data drift, and evaluate the performance of federated models across different clinical environments.</span>
              </li>
            </ul>
          </div>
        </section>

        {/* 4. Trust/Stat Chips (Optional, but good for credibility) */}
        <section className="flex flex-wrap justify-center gap-4 text-sm">
          <div className="px-4 py-2 bg-white border border-stonesense-line rounded-lg text-stonesense-ink/70 shadow-sm flex items-center gap-2">
            <span className="font-semibold text-stonesense-ink">98.5%</span> CT classification accuracy
          </div>
          <div className="px-4 py-2 bg-white border border-stonesense-line rounded-lg text-stonesense-ink/70 shadow-sm flex items-center gap-2">
            <span className="font-semibold text-stonesense-ink">91.7%</span> Risk model accuracy
          </div>
          <div className="px-4 py-2 bg-white border border-stonesense-line rounded-lg text-stonesense-ink/70 shadow-sm">
            Dual independent models
          </div>
          <div className="px-4 py-2 bg-white border border-stonesense-line rounded-lg text-stonesense-ink/70 shadow-sm">
            Explainable by design
          </div>
        </section>

        {/* 3. Features Section */}
        <section id="features" className="grid md:grid-cols-3 gap-6">
          <div className="bg-white p-8 rounded-2xl border border-stonesense-line shadow-sm hover:shadow-md transition-shadow">
            <div className="w-10 h-10 rounded-full bg-stonesense-teal/10 flex items-center justify-center mb-6">
              <svg className="w-5 h-5 text-stonesense-teal" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
            </div>
            <h3 className="font-serif text-xl mb-3 text-stonesense-ink">Submit patient cases</h3>
            <p className="text-stonesense-ink/70 text-sm leading-relaxed">
              Securely input urine biomarker data and upload CT scans for comprehensive AI evaluation.
            </p>
          </div>

          <div className="bg-white p-8 rounded-2xl border border-stonesense-line shadow-sm hover:shadow-md transition-shadow">
            <div className="w-10 h-10 rounded-full bg-stonesense-teal/10 flex items-center justify-center mb-6">
              <svg className="w-5 h-5 text-stonesense-teal" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
              </svg>
            </div>
            <h3 className="font-serif text-xl mb-3 text-stonesense-ink">Transparent AI results</h3>
            <p className="text-stonesense-ink/70 text-sm leading-relaxed">
              Get detailed risk scores and imaging classifications backed by SHAP feature contributions and Grad-CAM heatmaps.
            </p>
          </div>

          <div className="bg-white p-8 rounded-2xl border border-stonesense-line shadow-sm hover:shadow-md transition-shadow">
            <div className="w-10 h-10 rounded-full bg-stonesense-teal/10 flex items-center justify-center mb-6">
              <svg className="w-5 h-5 text-stonesense-teal" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
            </div>
            <h3 className="font-serif text-xl mb-3 text-stonesense-ink">Track and review</h3>
            <p className="text-stonesense-ink/70 text-sm leading-relaxed">
              Monitor patient history and review AI-assisted reports over time to support clinical workflows.
            </p>
          </div>
        </section>

        {/* 5. Sign-in Section (CTA) */}
        <section id="login" className="max-w-md mx-auto pt-8">
          <div className="mb-8 text-center">
            <h2 className="font-serif text-3xl">StoneSense</h2>
            <p className="text-sm text-stonesense-ink/60 mt-1">Sign in to continue</p>
          </div>

          {loginMode === "select" && (
            <div className="flex flex-col gap-3">
              <button
                onClick={() => setLoginMode("hospital")}
                className="rounded-lg border border-stonesense-line bg-white p-5 text-left hover:border-stonesense-teal transition-colors shadow-sm"
              >
                <p className="font-serif text-lg text-stonesense-ink">Hospital login</p>
                <p className="text-sm text-stonesense-ink/60 mt-1">
                  Submit patient cases and review risk / imaging results.
                </p>
              </button>

              <button
                onClick={() => setLoginMode("developer")}
                className="rounded-lg border border-stonesense-line bg-white p-5 text-left hover:border-stonesense-indigo transition-colors shadow-sm"
              >
                <p className="font-serif text-lg text-stonesense-ink">Developer login</p>
                <p className="text-sm text-stonesense-ink/60 mt-1">
                  Monitor model performance, deployments, and system health.
                </p>
              </button>
            </div>
          )}

          {loginMode === "hospital" && (
            <div className="rounded-lg border border-stonesense-line bg-white p-6 shadow-sm">
              <p className="font-serif text-lg mb-1 text-stonesense-ink">Hospital login</p>
              <p className="text-sm text-stonesense-ink/60 mb-4">
                Select your hospital to continue.
              </p>
              <select
                value={hospitalId ?? ""}
                onChange={(e) => setHospitalId(Number(e.target.value))}
                className="w-full rounded-md border border-stonesense-line px-3 py-2 text-sm mb-4 focus:outline-none focus:ring-1 focus:ring-stonesense-teal"
              >
                {hospitals.map((h) => (
                  <option key={h.id} value={h.id}>
                    {h.name}
                  </option>
                ))}
              </select>
              <button
                onClick={enterHospital}
                className="w-full rounded-md bg-stonesense-teal px-4 py-2 text-sm text-white hover:bg-stonesense-teal/90 transition-colors"
              >
                Continue
              </button>
              <button
                onClick={() => setLoginMode("select")}
                className="w-full mt-3 text-xs text-stonesense-ink/50 hover:text-stonesense-ink underline"
              >
                Back
              </button>
            </div>
          )}

          {loginMode === "developer" && (
            <div className="rounded-lg border border-stonesense-line bg-white p-6 shadow-sm">
              <p className="font-serif text-lg mb-1 text-stonesense-ink">Developer login</p>
              <p className="text-sm text-stonesense-ink/60 mb-4">
                No credentials required yet — access is open during this phase.
              </p>
              <button
                onClick={enterDeveloper}
                className="w-full rounded-md bg-stonesense-indigo px-4 py-2 text-sm text-white hover:bg-stonesense-indigo/90 transition-colors"
              >
                Continue
              </button>
              <button
                onClick={() => setLoginMode("select")}
                className="w-full mt-3 text-xs text-stonesense-ink/50 hover:text-stonesense-ink underline"
              >
                Back
              </button>
            </div>
          )}
        </section>
      </main>

      {/* 6. Footer */}
      <footer className="border-t border-stonesense-line bg-white py-8 mt-12">
        <div className="max-w-6xl mx-auto px-6 flex flex-col md:flex-row items-center justify-between gap-4 text-xs text-stonesense-ink/40">
          <p>© {new Date().getFullYear()} StoneSense-AI Contributors. MIT License.</p>
          <div className="flex gap-4">
            <a href="/privacy" className="hover:text-stonesense-ink transition-colors underline underline-offset-2">Privacy Policy</a>
            <p>Evidence-based support only — not a clinical diagnosis.</p>
          </div>
        </div>
      </footer>
    </div>
  );
}
