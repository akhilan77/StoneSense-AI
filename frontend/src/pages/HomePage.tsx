import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { ErrorMessage } from '../components/ErrorMessage';
import type { HealthResponse, ModelInfoResponse } from '../services/api';
import { getModelInfo, health } from '../services/api';

const summaryCards = [
  {
    route: '/risk-prediction',
    title: 'Risk Prediction',
    description: 'Submit patient measurements to receive risk classification scoring.',
  },
  {
    route: '/stone-detection',
    title: 'Stone Detection',
    description: 'Upload a CT or ultrasound image and inspect the detection details.',
  },
];

export function HomePage() {
  const [healthData, setHealthData] = useState<HealthResponse | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function loadStatus() {
      try {
        const h = await health();
        setHealthData(h);
        const m = await getModelInfo();
        setModelInfo(m);
      } catch (err) {
        setError('Backend services are currently offline.');
      }
    }
    loadStatus();
  }, []);

  return (
    <div className="space-y-6">
      <section className="rounded-3xl border border-slate-800 bg-gradient-to-br from-cyan-500/15 via-slate-900 to-slate-950 p-8 shadow-2xl shadow-slate-950/50">
        <div className="max-w-3xl">
          <p className="text-sm font-semibold uppercase tracking-[0.25em] text-cyan-300">
            StoneSense AI
          </p>
          <div className="mt-3 inline-flex items-center rounded-full border border-cyan-400/30 bg-cyan-400/10 px-3 py-1 text-sm font-medium text-cyan-200">
            UI polish • quick preview
          </div>
          <h1 className="mt-3 text-4xl font-bold text-white sm:text-5xl">
            Explainable AI-based kidney stone detection and risk prediction system
          </h1>
          <p className="mt-4 text-slate-300">
            This front-end consumes the FastAPI backend endpoints for health checks, patient risk
            scoring and independent image classification with transparent explanations.
          </p>
        </div>
      </section>

      {error ? <ErrorMessage message={error} /> : null}

      {/* System Status Dashboard */}
      {healthData && modelInfo ? (
        <section className="grid gap-5 md:grid-cols-2 lg:grid-cols-4">
          <div className="rounded-2xl border border-slate-850 bg-slate-900/60 p-5">
            <div className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
              Server Status
            </div>
            <div className="mt-2 text-lg font-bold text-emerald-400">Online</div>
          </div>
          <div className="rounded-2xl border border-slate-850 bg-slate-900/60 p-5">
            <div className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
              Device
            </div>
            <div className="mt-2 text-lg font-bold text-cyan-400 capitalize">
              {healthData.hardware.active_device}
            </div>
          </div>
          <div className="rounded-2xl border border-slate-850 bg-slate-900/60 p-5">
            <div className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
              CT ResNet18
            </div>
            <div className="mt-2 text-lg font-bold text-white">
              {modelInfo.dl_resnet18.accuracy * 100}% Accuracy
            </div>
          </div>
          <div className="rounded-2xl border border-slate-850 bg-slate-900/60 p-5">
            <div className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
              Risk XGBoost
            </div>
            <div className="mt-2 text-lg font-bold text-white">
              {modelInfo.ml_xgboost.validation_accuracy * 100}% Accuracy
            </div>
          </div>
        </section>
      ) : null}

      <section className="grid gap-4 md:grid-cols-3">
        {summaryCards.map((card) => (
          <Link
            key={card.route}
            to={card.route}
            className="rounded-3xl border border-slate-800 bg-slate-900/70 p-5 transition duration-200 hover:-translate-y-1 hover:border-cyan-400 hover:bg-slate-900"
          >
            <div className="text-lg font-semibold text-white">{card.title}</div>
            <p className="mt-2 text-sm text-slate-400">{card.description}</p>
          </Link>
        ))}
      </section>
    </div>
  );
}
