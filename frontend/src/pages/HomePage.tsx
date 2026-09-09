import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { ErrorMessage } from '../components/ErrorMessage';
import type { HealthResponse, ModelInfoResponse } from '../services/api';
import { getModelInfo, health } from '../services/api';

const summaryCards = [
  {
    route: '/hospital-dashboard',
    title: 'Hospital Dashboard',
    description: 'Submit patient cases, urine biomarkers, and review independent ML/DL evidence.',
  },
  {
    route: '/developer-dashboard',
    title: 'Developer Console',
    description: 'Monitor aggregate system health, model deployments, logs, and drift analysis.',
  },
  {
    route: '/risk-prediction',
    title: 'Risk Prediction',
    description: 'Submit patient measurements to receive risk classification scoring.',
  },
  {
    route: '/stone-detection',
    title: 'Stone Detection',
    description: 'Upload a CT scan image and inspect the detection details with Grad-CAM.',
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
      <section className="rounded-xl border border-stonesense-line bg-white p-8 shadow-sm">
        <div className="max-w-3xl">
          <p className="text-xs font-semibold uppercase tracking-[0.25em] text-stonesense-teal">
            StoneSense AI
          </p>
          <h1 className="mt-2 font-serif text-3xl font-semibold text-stonesense-ink sm:text-4xl">
            Explainable AI-based kidney stone detection and risk prediction system
          </h1>
          <p className="mt-3 text-stonesense-ink/70">
            Multi-hospital clinical evidence system consuming FastAPI backend endpoints for health checks, patient risk
            scoring, and independent image classification with transparent explanations.
          </p>
          <div className="mt-6 flex gap-3">
            <Link
              to="/login"
              className="rounded-md bg-stonesense-teal px-4 py-2 text-sm font-medium text-white shadow-sm hover:bg-stonesense-teal/90"
            >
              Go to Login
            </Link>
            <Link
              to="/hospital-dashboard"
              className="rounded-md border border-stonesense-line bg-white px-4 py-2 text-sm font-medium text-stonesense-ink hover:bg-stonesense-ink/5"
            >
              Hospital Console
            </Link>
          </div>
        </div>
      </section>

      {error ? <ErrorMessage message={error} /> : null}

      {/* System Status Dashboard */}
      {healthData && modelInfo ? (
        <section className="grid gap-5 md:grid-cols-2 lg:grid-cols-4">
          <div className="rounded-xl border border-stonesense-line bg-white p-5 shadow-sm">
            <div className="text-xs font-medium uppercase tracking-wider text-stonesense-ink/50">
              Server Status
            </div>
            <div className="mt-2 text-lg font-bold text-stonesense-teal">Online</div>
          </div>
          <div className="rounded-xl border border-stonesense-line bg-white p-5 shadow-sm">
            <div className="text-xs font-medium uppercase tracking-wider text-stonesense-ink/50">
              Device
            </div>
            <div className="mt-2 text-lg font-bold text-stonesense-indigo capitalize">
              {healthData.hardware.active_device}
            </div>
          </div>
          <div className="rounded-xl border border-stonesense-line bg-white p-5 shadow-sm">
            <div className="text-xs font-medium uppercase tracking-wider text-stonesense-ink/50">
              CT ResNet18
            </div>
            <div className="mt-2 text-lg font-bold text-stonesense-ink">
              {(modelInfo.dl_resnet18.accuracy * 100).toFixed(1)}% Accuracy
            </div>
          </div>
          <div className="rounded-xl border border-stonesense-line bg-white p-5 shadow-sm">
            <div className="text-xs font-medium uppercase tracking-wider text-stonesense-ink/50">
              Risk XGBoost
            </div>
            <div className="mt-2 text-lg font-bold text-stonesense-ink">
              {(modelInfo.ml_xgboost.validation_accuracy * 100).toFixed(1)}% Accuracy
            </div>
          </div>
        </section>
      ) : null}

      <section className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        {summaryCards.map((card) => (
          <Link
            key={card.route}
            to={card.route}
            className="rounded-xl border border-stonesense-line bg-white p-5 shadow-sm transition hover:border-stonesense-teal hover:shadow-md"
          >
            <div className="font-serif text-lg font-semibold text-stonesense-ink">{card.title}</div>
            <p className="mt-2 text-sm text-stonesense-ink/60">{card.description}</p>
          </Link>
        ))}
      </section>
    </div>
  );
}
