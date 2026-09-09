// Drop at: frontend/src/pages/HospitalDashboard.tsx
import { useEffect, useState } from "react";
import AppLayout from "../components/layout/AppLayout";
import { useHospital } from "../context/HospitalContext";
import { fetchHistory } from "../services/hospitalApi";
import { PatientHistoryItem } from "../types/dashboard";

const steps = [
  { key: "input", label: "Patient input" },
  { key: "results", label: "ML & DL results" },
  { key: "explain", label: "Explainability" },
  { key: "assessment", label: "Trustworthy assessment" },
] as const;

export default function HospitalDashboard() {
  const { hospitalId } = useHospital();
  const [history, setHistory] = useState<PatientHistoryItem[]>([]);
  const [activeStep, setActiveStep] = useState<(typeof steps)[number]["key"]>("input");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!hospitalId) return;
    setLoading(true);
    fetchHistory(hospitalId)
      .then(setHistory)
      .catch(() => setHistory([]))
      .finally(() => setLoading(false));
  }, [hospitalId]);

  return (
    <AppLayout
      role="hospital"
      title="Patient workflow"
      subtitle="Independent clinical and imaging evidence, reviewed together — not a diagnosis."
    >
      {/* Stepped workflow — genuinely sequential, so numbering is earned here */}
      <ol className="flex items-center gap-2 mb-8">
        {steps.map((step, i) => (
          <li key={step.key} className="flex items-center gap-2">
            <button
              onClick={() => setActiveStep(step.key)}
              className={`flex items-center gap-2 rounded-full border px-4 py-1.5 text-sm transition-colors ${
                activeStep === step.key
                  ? "border-stonesense-teal bg-stonesense-teal text-white"
                  : "border-stonesense-line text-stonesense-ink/60 hover:border-stonesense-teal/50"
              }`}
            >
              <span className="text-xs opacity-70">{i + 1}</span>
              {step.label}
            </button>
            {i < steps.length - 1 && <span className="h-px w-6 bg-stonesense-line" />}
          </li>
        ))}
      </ol>

      <div className="grid grid-cols-3 gap-6">
        <section className="col-span-2 rounded-lg border border-stonesense-line bg-white p-6 min-h-[420px]">
          {activeStep === "input" && (
            <div>
              <h2 className="font-serif text-lg mb-1">Patient input</h2>
              <p className="text-sm text-stonesense-ink/60 mb-4">
                Urine biomarkers and CT scan slice for this patient.
              </p>
              <div className="grid grid-cols-2 gap-3">
                {["Specific gravity", "pH", "Osmolality", "Conductivity", "Urea", "Calcium"].map((f) => (
                  <label key={f} className="text-sm">
                    <span className="block text-stonesense-ink/60 mb-1">{f}</span>
                    <input className="w-full rounded-md border border-stonesense-line px-3 py-1.5" />
                  </label>
                ))}
              </div>
              <label className="mt-4 block rounded-md border border-dashed border-stonesense-line p-6 text-center text-sm text-stonesense-ink/50 cursor-pointer">
                Drop a CT scan slice here (DICOM / PNG / JPG)
                <input type="file" className="hidden" />
              </label>
              <button className="mt-4 rounded-md bg-stonesense-teal px-4 py-2 text-sm text-white">
                Run assessment
              </button>
            </div>
          )}

          {activeStep === "results" && (
            <div className="grid grid-cols-2 gap-5">
              <div>
                <h3 className="font-serif text-base mb-2">ML result — risk score</h3>
                <p className="text-3xl font-semibold text-stonesense-teal">Moderate</p>
                <p className="text-sm text-stonesense-ink/60 mt-1">
                  Top risk factors: low pH, elevated calcium.
                </p>
              </div>
              <div>
                <h3 className="font-serif text-base mb-2">DL result — stone detection</h3>
                <p className="text-3xl font-semibold text-stonesense-amber">Stone detected</p>
                <p className="text-sm text-stonesense-ink/60 mt-1">
                  Confidence 94.2% · Location: left kidney, lower pole
                </p>
              </div>
            </div>
          )}

          {activeStep === "explain" && (
            <div className="grid grid-cols-2 gap-5">
              <div>
                <h3 className="font-serif text-base mb-2">SHAP — biomarker contribution</h3>
                <div className="h-48 rounded-md bg-stonesense-paper flex items-center justify-center text-sm text-stonesense-ink/40">
                  SHAP waterfall chart
                </div>
              </div>
              <div>
                <h3 className="font-serif text-base mb-2">Grad-CAM++ — CT heatmap</h3>
                <div className="h-48 rounded-md bg-stonesense-paper flex items-center justify-center text-sm text-stonesense-ink/40">
                  CT slice with saliency overlay
                </div>
              </div>
            </div>
          )}

          {activeStep === "assessment" && (
            <div>
              <h3 className="font-serif text-base mb-3">Trustworthy assessment</h3>
              <div className="rounded-md border border-stonesense-line p-4 text-sm">
                <p>
                  <span className="text-stonesense-ink/60">Clinical model:</span> Moderate risk
                </p>
                <p>
                  <span className="text-stonesense-ink/60">Imaging model:</span> Stone detected
                </p>
                <p className="mt-2 text-stonesense-ink/70">
                  Both evidence streams agree a stone is likely present. This is a transparent
                  rule-based comparison of two independent models — not a fused prediction.
                </p>
              </div>
              <button className="mt-4 rounded-md border border-stonesense-line px-4 py-2 text-sm">
                Download report
              </button>
            </div>
          )}
        </section>

        <section className="rounded-lg border border-stonesense-line bg-white p-6">
          <h2 className="font-serif text-lg mb-3">Recent cases</h2>
          {loading ? (
            <p className="text-sm text-stonesense-ink/40">Loading…</p>
          ) : history.length === 0 ? (
            <p className="text-sm text-stonesense-ink/40">No cases recorded yet for this hospital.</p>
          ) : (
            <ul className="flex flex-col gap-3">
              {history.map((item) => (
                <li key={item.id} className="text-sm border-b border-stonesense-line pb-2">
                  <p className="font-medium">{item.reference_code}</p>
                  <p className="text-stonesense-ink/60">
                    {item.model_name} · {item.result_label ?? "—"}
                    {item.confidence != null && ` · ${(item.confidence * 100).toFixed(1)}%`}
                  </p>
                </li>
              ))}
            </ul>
          )}
        </section>
      </div>
    </AppLayout>
  );
}
