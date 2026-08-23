import { PatientForm } from '../components/PatientForm';

export function RiskPredictionPage() {
  return (
    <div className="space-y-6">
      <section className="rounded-3xl border border-slate-800 bg-slate-900/50 p-5">
        <h1 className="text-2xl font-semibold text-white">
          Clinical Risk Prediction &amp; Trustworthy Assessment
        </h1>
        <p className="mt-2 text-sm text-slate-400">
          Review the XGBoost clinical-risk evidence and its SHAP explanation independently from
          imaging.
        </p>
      </section>
      <PatientForm />
    </div>
  );
}
