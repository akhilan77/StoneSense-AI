export function PrivacyPolicy() {
  return (
    <div className="min-h-screen bg-stonesense-paper text-stonesense-ink font-sans selection:bg-stonesense-teal selection:text-white">
      {/* Header */}
      <header className="border-b border-stonesense-ink/10 bg-stonesense-paper sticky top-0 z-10">
        <div className="h-1 bg-stonesense-ink w-full"></div>
        <div className="max-w-6xl mx-auto px-6 h-16 flex items-center justify-between">
          <a href="/" className="font-serif text-2xl font-medium tracking-tight hover:text-stonesense-teal transition-colors">StoneSense-AI</a>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-3xl mx-auto px-6 py-16 md:py-24 space-y-12">
        <div className="space-y-4">
          <h1 className="font-serif text-4xl font-medium text-stonesense-ink">Privacy Policy</h1>
          <p className="text-stonesense-ink/60">Last updated: {new Date().toLocaleDateString()}</p>
        </div>

        <section className="space-y-4">
          <h2 className="font-serif text-2xl text-stonesense-ink">1. Data Collection and Usage</h2>
          <p className="text-stonesense-ink/70 leading-relaxed">
            StoneSense-AI processes clinical inputs, including urine biomarkers and CT scan imagery. In a production environment, all data processed by StoneSense-AI models must be fully de-identified and stripped of Protected Health Information (PHI) prior to submission.
          </p>
        </section>

        <section className="space-y-4">
          <h2 className="font-serif text-2xl text-stonesense-ink">2. Federated Learning Infrastructure</h2>
          <p className="text-stonesense-ink/70 leading-relaxed">
            Our platform supports a federated AI architecture. In this mode, raw patient data never leaves the hospital's local environment. Only aggregated, anonymized model weights and gradients are transmitted to our central server for global model optimization.
          </p>
        </section>

        <section className="space-y-4">
          <h2 className="font-serif text-2xl text-stonesense-ink">3. Explainability and Logging</h2>
          <p className="text-stonesense-ink/70 leading-relaxed">
            To ensure model transparency, system-level metrics (e.g., drift scores, hardware utilization, classification confidence intervals) are recorded. These logs do not contain raw patient imagery or identifying details.
          </p>
        </section>

        <section className="space-y-4">
          <h2 className="font-serif text-2xl text-stonesense-ink">4. HIPAA & GDPR Compliance</h2>
          <p className="text-stonesense-ink/70 leading-relaxed">
            Note: This instance of StoneSense-AI is a demonstration and decision-support prototype. Before clinical deployment, organizations must execute a Business Associate Agreement (BAA) and conduct a comprehensive security review to meet HIPAA (US) or GDPR (EU) standards.
          </p>
        </section>
        
        <div className="pt-8">
          <a href="/" className="inline-flex items-center gap-2 px-6 py-3 rounded-lg bg-stonesense-teal text-white font-medium hover:bg-stonesense-teal/90 transition-colors">
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" /></svg>
            Back to Home
          </a>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-stonesense-line bg-white py-8 mt-12">
        <div className="max-w-6xl mx-auto px-6 flex flex-col md:flex-row items-center justify-between gap-4 text-xs text-stonesense-ink/40">
          <p>© {new Date().getFullYear()} StoneSense-AI Contributors. MIT License.</p>
          <p>Evidence-based support only — not a clinical diagnosis.</p>
        </div>
      </footer>
    </div>
  );
}
