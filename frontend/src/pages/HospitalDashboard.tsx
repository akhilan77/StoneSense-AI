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

  // Form input states pre-filled to match visual reference (Image 1)
  const [gravity, setGravity] = useState("1.020");
  const [ph, setPh] = useState("5.8");
  const [osmolality, setOsmolality] = useState("620");
  const [conductivity, setConductivity] = useState("24.1");
  const [urea, setUrea] = useState("380");
  const [calcium, setCalcium] = useState("4.5");

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
      {/* Stepped workflow */}
      <ol className="flex items-center gap-2 mb-6">
        {steps.map((step, i) => {
          const isActive = activeStep === step.key;
          return (
            <li key={step.key} className="flex items-center gap-2">
              <button
                onClick={() => setActiveStep(step.key)}
                className={`step-btn flex items-center gap-2 rounded-full border px-4 py-1.5 text-xs transition-colors cursor-pointer ${
                  isActive
                    ? "active bg-[#1F6F5C] border-[#1F6F5C] text-white font-medium"
                    : "bg-transparent border-[#DDE3DC] text-[#101B16]/60 hover:border-[#1F6F5C]/50"
                }`}
              >
                <span className="text-[11px] opacity-75">{i + 1}</span>
                {step.label}
              </button>
              {i < steps.length - 1 && <span className="h-px w-5 bg-[#DDE3DC]" />}
            </li>
          );
        })}
      </ol>

      <div className="grid grid-cols-3 gap-6">
        <section className="col-span-2 rounded-xl border border-[#DDE3DC] bg-white p-6 min-h-[420px] shadow-xs">
          {activeStep === "input" && (
            <div>
              <h2 className="font-serif text-lg font-medium text-[#101B16] mb-1">Patient input</h2>
              <p className="text-xs text-[#101B16]/55 mb-4">
                Urine biomarkers and CT scan slice for this patient.
              </p>
              <div className="grid grid-cols-2 gap-3">
                <label className="text-xs text-[#101B16]/60">
                  <span className="block mb-1">Specific gravity</span>
                  <input
                    value={gravity}
                    onChange={(e) => setGravity(e.target.value)}
                    className="w-full rounded-md border border-[#DDE3DC] bg-white px-3 py-1.5 text-sm text-[#101B16] outline-none focus:border-[#1F6F5C]"
                  />
                </label>
                <label className="text-xs text-[#101B16]/60">
                  <span className="block mb-1">pH</span>
                  <input
                    value={ph}
                    onChange={(e) => setPh(e.target.value)}
                    className="w-full rounded-md border border-[#DDE3DC] bg-white px-3 py-1.5 text-sm text-[#101B16] outline-none focus:border-[#1F6F5C]"
                  />
                </label>
                <label className="text-xs text-[#101B16]/60">
                  <span className="block mb-1">Osmolality</span>
                  <input
                    value={osmolality}
                    onChange={(e) => setOsmolality(e.target.value)}
                    className="w-full rounded-md border border-[#DDE3DC] bg-white px-3 py-1.5 text-sm text-[#101B16] outline-none focus:border-[#1F6F5C]"
                  />
                </label>
                <label className="text-xs text-[#101B16]/60">
                  <span className="block mb-1">Conductivity</span>
                  <input
                    value={conductivity}
                    onChange={(e) => setConductivity(e.target.value)}
                    className="w-full rounded-md border border-[#DDE3DC] bg-white px-3 py-1.5 text-sm text-[#101B16] outline-none focus:border-[#1F6F5C]"
                  />
                </label>
                <label className="text-xs text-[#101B16]/60">
                  <span className="block mb-1">Urea</span>
                  <input
                    value={urea}
                    onChange={(e) => setUrea(e.target.value)}
                    className="w-full rounded-md border border-[#DDE3DC] bg-white px-3 py-1.5 text-sm text-[#101B16] outline-none focus:border-[#1F6F5C]"
                  />
                </label>
                <label className="text-xs text-[#101B16]/60">
                  <span className="block mb-1">Calcium</span>
                  <input
                    value={calcium}
                    onChange={(e) => setCalcium(e.target.value)}
                    className="w-full rounded-md border border-[#DDE3DC] bg-white px-3 py-1.5 text-sm text-[#101B16] outline-none focus:border-[#1F6F5C]"
                  />
                </label>
              </div>
              <label className="mt-4 block rounded-lg border border-dashed border-[#DDE3DC] p-6 text-center text-xs text-[#101B16]/45 cursor-pointer hover:border-[#1F6F5C]/50 transition-colors">
                Drop a CT scan slice here (DICOM / PNG / JPG)
                <input type="file" className="hidden" />
              </label>
              <button
                onClick={() => setActiveStep("results")}
                className="btn-primary mt-4 rounded-md bg-[#1F6F5C] px-4 py-2 text-xs font-medium text-white hover:bg-[#185849] cursor-pointer"
              >
                Run assessment
              </button>
            </div>
          )}

          {activeStep === "results" && (
            <div className="grid grid-cols-2 gap-5">
              <div>
                <h3 className="font-serif text-base font-medium text-[#101B16] mb-2">ML result — risk score</h3>
                <p className="text-3xl font-semibold text-[#1F6F5C]">Moderate</p>
                <p className="text-xs text-[#101B16]/60 mt-1">
                  Top risk factors: low pH, elevated calcium.
                </p>
              </div>
              <div>
                <h3 className="font-serif text-base font-medium text-[#101B16] mb-2">DL result — stone detection</h3>
                <p className="text-3xl font-semibold text-[#C97A2B]">Stone detected</p>
                <p className="text-xs text-[#101B16]/60 mt-1">
                  Confidence 94.2% · Location: left kidney, lower pole
                </p>
              </div>
            </div>
          )}

          {activeStep === "explain" && (
            <div className="grid grid-cols-2 gap-5">
              <div>
                <h3 className="font-serif text-base font-medium text-[#101B16] mb-2">SHAP — biomarker contribution</h3>
                <div className="h-48 rounded-md bg-[#F3F6F1] flex items-center justify-center text-xs text-[#101B16]/40">
                  SHAP waterfall chart
                </div>
              </div>
              <div>
                <h3 className="font-serif text-base font-medium text-[#101B16] mb-2">Grad-CAM++ — CT heatmap</h3>
                <div className="h-48 rounded-md bg-[#F3F6F1] flex items-center justify-center text-xs text-[#101B16]/40">
                  CT slice with saliency overlay
                </div>
              </div>
            </div>
          )}

          {activeStep === "assessment" && (
            <div>
              <h3 className="font-serif text-base font-medium text-[#101B16] mb-3">Trustworthy assessment</h3>
              <div className="rounded-md border border-[#DDE3DC] p-4 text-xs">
                <p>
                  <span className="text-[#101B16]/60">Clinical model:</span> Moderate risk
                </p>
                <p className="mt-1">
                  <span className="text-[#101B16]/60">Imaging model:</span> Stone detected
                </p>
                <p className="mt-2 text-[#101B16]/70 leading-relaxed">
                  Both evidence streams agree a stone is likely present. This is a transparent
                  rule-based comparison of two independent models — not a fused prediction.
                </p>
              </div>
              <button className="mt-4 rounded-md border border-[#DDE3DC] bg-white px-4 py-2 text-xs text-[#101B16]/80 hover:bg-black/5 cursor-pointer">
                Download report
              </button>
            </div>
          )}
        </section>

        <section className="rounded-xl border border-[#DDE3DC] bg-white p-6 shadow-xs">
          <h2 className="font-serif text-lg font-medium text-[#101B16] mb-3">Recent cases</h2>
          {loading ? (
            <p className="text-xs text-[#101B16]/40">Loading…</p>
          ) : history.length === 0 ? (
            <p className="text-xs text-[#101B16]/40">No cases recorded yet for this hospital.</p>
          ) : (
            <ul className="flex flex-col gap-3">
              {history.map((item, idx) => {
                const refCode = item.reference_code && item.reference_code !== "—" 
                  ? item.reference_code 
                  : `PT-${2201 - idx}`;
                return (
                  <li key={item.id} className="border-b border-[#DDE3DC] pb-2.5">
                    <p className="text-xs font-semibold text-[#101B16]">{refCode}</p>
                    <p className="text-[12px] text-[#101B16]/55 mt-0.5">
                      {item.model_name} · {item.result_label ?? "—"}
                      {item.confidence != null && ` · ${(item.confidence * 100).toFixed(1)}%`}
                    </p>
                  </li>
                );
              })}
            </ul>
          )}
        </section>
      </div>
    </AppLayout>
  );
}
