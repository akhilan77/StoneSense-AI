import { Link } from 'react-router-dom';

const summaryCards = [
  { route: '/risk-prediction', title: 'Risk Prediction', description: 'Submit patient measurements to receive a mock risk classification.' },
  { route: '/stone-detection', title: 'Stone Detection', description: 'Upload a CT or ultrasound image and inspect the mock detection payload.' },
  { route: '/assessment', title: 'Assessment', description: 'Compose risk, image, and explainability into a final care recommendation.' },
];

export function HomePage() {
  return (
    <div className="space-y-6">
      <section className="rounded-3xl border border-slate-800 bg-gradient-to-br from-cyan-500/15 via-slate-900 to-slate-950 p-8 shadow-2xl shadow-slate-950/50">
        <div className="max-w-3xl">
          <p className="text-sm font-semibold uppercase tracking-[0.25em] text-cyan-300">StoneSense AI</p>
          <h1 className="mt-3 text-4xl font-bold text-white sm:text-5xl">Explainable AI-based kidney stone detection and risk prediction system</h1>
          <p className="mt-4 text-slate-300">This front-end consumes the existing FastAPI mock backend contract for health checks, patient risk scoring, image detection, and full assessment feedback.</p>
        </div>
      </section>

      <section className="grid gap-4 md:grid-cols-3">
        {summaryCards.map((card) => (
          <Link key={card.route} to={card.route} className="rounded-3xl border border-slate-800 bg-slate-900/70 p-5 transition hover:border-cyan-400 hover:bg-slate-900">
            <div className="text-lg font-semibold text-white">{card.title}</div>
            <p className="mt-2 text-sm text-slate-400">{card.description}</p>
          </Link>
        ))}
      </section>
    </div>
  );
}
