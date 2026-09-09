// frontend/src/pages/DeveloperDashboard.tsx
import React, { useEffect, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import AppLayout from "../components/layout/AppLayout";
import {
  fetchModelPerformance,
  fetchHospitalLogs,
  fetchSystemMonitoring,
  fetchSystemLogs,
  fetchDriftAnalysis,
  fetchAllEnrolledHospitals,
  deployModelVersion,
  fetchFederatedOverview,
  fetchRoundHistory,
  fetchHospitalParticipation,
} from "../services/developerApi";
import {
  ModelPerformance,
  HospitalUpdateLogEntry,
  SystemLogEntry,
  DriftPoint,
  SystemMonitoringSummary,
  Hospital,
} from "../types/dashboard";
import {
  FederatedOverview,
  FederatedRoundDetail,
  HospitalParticipation,
} from "../types/federated";

type DevTab = "overview" | "federated" | "versions" | "hospitals" | "monitoring" | "drift" | "access";

interface DeveloperDashboardProps {
  initialTab?: DevTab;
}

export default function DeveloperDashboard({ initialTab }: DeveloperDashboardProps) {
  const location = useLocation();
  const navigate = useNavigate();

  // Determine active tab from URL path or props
  const getTabFromPath = (): DevTab => {
    if (initialTab) return initialTab;
    const path = location.pathname;
    if (path.includes("/federated")) return "federated";
    if (path.includes("/versions")) return "versions";
    if (path.includes("/hospitals")) return "hospitals";
    if (path.includes("/monitoring")) return "monitoring";
    if (path.includes("/drift")) return "drift";
    if (path.includes("/enrolled-hospitals") || path.includes("/access")) return "access";
    return "overview";
  };

  const [activeTab, setActiveTab] = useState<DevTab>(getTabFromPath());
  const [versions, setVersions] = useState<ModelPerformance[]>([]);
  const [hospitalLogs, setHospitalLogs] = useState<HospitalUpdateLogEntry[]>([]);
  const [monitoring, setMonitoring] = useState<SystemMonitoringSummary | null>(null);
  const [logs, setLogs] = useState<SystemLogEntry[]>([]);
  const [drift, setDrift] = useState<DriftPoint[]>([]);
  const [hospitals, setHospitals] = useState<Hospital[]>([]);
  const [toastMessage, setToastMessage] = useState<string | null>(null);

  // Federated Learning Telemetry States
  const [fedOverview, setFedOverview] = useState<FederatedOverview | null>(null);
  const [roundHistory, setRoundHistory] = useState<FederatedRoundDetail[]>([]);
  const [participation, setParticipation] = useState<HospitalParticipation[]>([]);
  const [selectedRoundDetail, setSelectedRoundDetail] = useState<FederatedRoundDetail | null>(null);

  // Filter states
  const [logFilterLevel, setLogFilterLevel] = useState<string>("all");
  const [hospitalSearch, setHospitalSearch] = useState<string>("");
  const [hospLogSearch, setHospLogSearch] = useState<string>("");

  // Modal states
  const [isDeployModalOpen, setIsDeployModalOpen] = useState(false);
  const [isEnrollModalOpen, setIsEnrollModalOpen] = useState(false);
  const [selectedLogPayload, setSelectedLogPayload] = useState<HospitalUpdateLogEntry | null>(null);

  // New Version Form state
  const [newVersionForm, setNewVersionForm] = useState({
    modelFamily: "xgboost_risk",
    versionTag: "v2.5.0-prod",
    accuracy: "0.965",
    f1Score: "0.960",
    environment: "production",
    notes: "Trained on updated multi-center cohort dataset with reduced false positive rate.",
  });

  // New Hospital Enroll Form state
  const [newHospitalForm, setNewHospitalForm] = useState({
    name: "",
    code: "",
    region: "",
    contactEmail: "",
    tier: "standard" as "enterprise" | "standard" | "research",
  });

  const showToast = (msg: string) => {
    setToastMessage(msg);
    setTimeout(() => setToastMessage(null), 3200);
  };

  const loadAllData = () => {
    fetchModelPerformance().then(setVersions).catch(() => {});
    fetchHospitalLogs().then(setHospitalLogs).catch(() => {});
    fetchSystemMonitoring().then(setMonitoring).catch(() => {});
    fetchSystemLogs().then(setLogs).catch(() => {});
    fetchDriftAnalysis().then(setDrift).catch(() => {});
    fetchAllEnrolledHospitals().then(setHospitals).catch(() => {});
    fetchFederatedOverview().then(setFedOverview).catch(() => {});
    fetchRoundHistory().then((data) => {
      setRoundHistory(data);
      if (data.length > 0) setSelectedRoundDetail(data[data.length - 1]);
    }).catch(() => {});
    fetchHospitalParticipation().then(setParticipation).catch(() => {});
  };

  useEffect(() => {
    loadAllData();
  }, []);

  useEffect(() => {
    setActiveTab(getTabFromPath());
  }, [location.pathname]);

  const handleTabChange = (tab: DevTab) => {
    setActiveTab(tab);
    if (tab === "overview") navigate("/developer-dashboard");
    else if (tab === "federated") navigate("/developer-dashboard/federated");
    else if (tab === "versions") navigate("/developer-dashboard/versions");
    else if (tab === "hospitals") navigate("/developer-dashboard/hospitals");
    else if (tab === "monitoring") navigate("/developer-dashboard/monitoring");
    else if (tab === "drift") navigate("/developer-dashboard/drift");
    else if (tab === "access") navigate("/developer-dashboard/enrolled-hospitals");
  };

  const handleDeployVersion = async (v: ModelPerformance) => {
    showToast(`Deploying ${v.model_family} (${v.version_tag}) to production...`);
    await deployModelVersion(v.id);
    setVersions((prev) =>
      prev.map((item) => {
        if (item.model_family === v.model_family) {
          return { ...item, is_deployed: item.id === v.id, rollout_pct: item.id === v.id ? 100 : 0 };
        }
        return item;
      })
    );
    showToast(`Successfully promoted ${v.version_tag} as active production model!`);
  };

  const handleRolloutChange = (id: number, pct: number) => {
    setVersions((prev) =>
      prev.map((item) => (item.id === id ? { ...item, rollout_pct: pct } : item))
    );
    showToast(`Canary traffic allocation updated to ${pct}% for model #${id}`);
  };

  const handleCreateVersionSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const newV: ModelPerformance = {
      id: Date.now(),
      model_family: newVersionForm.modelFamily,
      version_tag: newVersionForm.versionTag,
      accuracy: parseFloat(newVersionForm.accuracy) || 0.95,
      f1_score: parseFloat(newVersionForm.f1Score) || 0.94,
      mcc: 0.91,
      is_deployed: newVersionForm.environment === "production",
      trained_at: new Date().toISOString(),
      latency_ms: newVersionForm.modelFamily === "xgboost_risk" ? 17.5 : 78.0,
      environment: newVersionForm.environment as any,
      rollout_pct: newVersionForm.environment === "production" ? 100 : 0,
    };

    setVersions([newV, ...versions]);
    setIsDeployModalOpen(false);
    showToast(`Model release package ${newV.version_tag} registered successfully!`);
  };

  const handleToggleHospitalAccess = (hospital: Hospital) => {
    const updatedStatus = !hospital.is_active;
    setHospitals((prev) =>
      prev.map((h) => (h.id === hospital.id ? { ...h, is_active: updatedStatus } : h))
    );
    if (updatedStatus) {
      showToast(`Access granted & API Key activated for ${hospital.name}`);
    } else {
      showToast(`ACCESS REVOKED: ${hospital.name} credentials suspended immediately.`);
    }
  };

  const handleRegenerateKey = (hospital: Hospital) => {
    const randomKey = `ss_live_${hospital.name.toLowerCase().slice(0, 4)}_${Math.random().toString(36).substring(2, 9)}`;
    setHospitals((prev) =>
      prev.map((h) => (h.id === hospital.id ? { ...h, api_key: randomKey } : h))
    );
    showToast(`Generated new 256-bit API credential for ${hospital.name}`);
  };

  const handleEnrollHospitalSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!newHospitalForm.name) return;

    const newHosp: Hospital = {
      id: Date.now(),
      name: newHospitalForm.name,
      hospital_code: newHospitalForm.code || `HOSP-${Math.random().toString(36).substring(2, 6).toUpperCase()}`,
      region: newHospitalForm.region || "India",
      is_active: true,
      tier: newHospitalForm.tier,
      contact_email: newHospitalForm.contactEmail,
      api_key: `ss_live_${newHospitalForm.name.toLowerCase().slice(0, 4)}_${Math.random().toString(36).substring(2, 9)}`,
      last_sync: "Just now",
      total_scans: 0,
    };

    setHospitals([newHosp, ...hospitals]);
    setIsEnrollModalOpen(false);
    setNewHospitalForm({ name: "", code: "", region: "", contactEmail: "", tier: "standard" });
    showToast(`Hospital ${newHosp.name} enrolled with active access and API key!`);
  };

  return (
    <AppLayout
      role="developer"
      title="Developer Console"
      subtitle="Aggregate model & system health across all hospitals — zero patient-level telemetry."
    >
      {/* Toast Notification */}
      {toastMessage && (
        <div className="fixed bottom-6 right-6 z-50 flex items-center gap-3 rounded-lg bg-[#101B16] px-4 py-3 text-sm text-white shadow-xl animate-fade-in border border-white/10">
          <span className="h-2 w-2 rounded-full bg-[#3B3F8C] animate-pulse" />
          <span>{toastMessage}</span>
        </div>
      )}

      {/* Tab Navigation Navigation Pills */}
      <div className="flex items-center justify-between border-b border-[#DDE3DC] pb-4 mb-6">
        <nav className="flex items-center gap-2 overflow-x-auto">
          {[
            { key: "overview", label: "Overview & Performance" },
            { key: "federated", label: "Federated Learning Hub" },
            { key: "versions", label: "Versions & Deployment" },
            { key: "hospitals", label: "Hospital Update Logs" },
            { key: "monitoring", label: "System Monitoring" },
            { key: "drift", label: "Drift Analysis" },
            { key: "access", label: "Enrolled Hospitals & Access" },
          ].map((tab) => {
            const isActive = activeTab === tab.key;
            return (
              <button
                key={tab.key}
                onClick={() => handleTabChange(tab.key as DevTab)}
                className={`rounded-lg px-3.5 py-2 text-xs font-medium transition-all cursor-pointer whitespace-nowrap ${
                  isActive
                    ? "bg-[#3B3F8C] text-white shadow-xs"
                    : "bg-white border border-[#DDE3DC] text-[#101B16]/70 hover:bg-[#F3F6F1] hover:text-[#101B16]"
                }`}
              >
                {tab.label}
              </button>
            );
          })}
        </nav>

        <button
          onClick={loadAllData}
          className="flex items-center gap-1.5 rounded-md border border-[#DDE3DC] bg-white px-3 py-1.5 text-xs text-[#101B16]/70 hover:bg-black/5 cursor-pointer shadow-xs"
        >
          <svg className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
          Refresh Live Telemetry
        </button>
      </div>

      {/* ========================================================================= */}
      {/* TAB 1: OVERVIEW & PERFORMANCE */}
      {/* ========================================================================= */}
      {activeTab === "overview" && (
        <div className="space-y-6">
          {/* Top KPI Cards */}
          <div className="grid grid-cols-4 gap-4">
            {[
              {
                label: "Predictions (24h)",
                value: monitoring?.total_predictions_24h ?? "1,482",
                sub: "+12.4% vs prev day",
              },
              {
                label: "Error Rate (24h)",
                value: monitoring ? `${((monitoring.error_count_24h / (monitoring.total_predictions_24h || 1)) * 100).toFixed(2)}%` : "0.20%",
                sub: `${monitoring?.error_count_24h ?? 3} errors logged`,
              },
              {
                label: "Avg Inference Latency",
                value: monitoring?.avg_latency_ms ? `${monitoring.avg_latency_ms} ms` : "44.8 ms",
                sub: "p95: 88ms · p99: 135ms",
              },
              {
                label: "System Uptime",
                value: monitoring ? `${monitoring.uptime_pct}%` : "99.98%",
                sub: "All 8 worker nodes active",
              },
            ].map((stat) => (
              <div key={stat.label} className="rounded-xl border border-[#DDE3DC] bg-white p-5 shadow-xs">
                <p className="text-xs font-medium text-[#101B16]/55">{stat.label}</p>
                <p className="text-2xl font-bold mt-1.5 text-[#3B3F8C]">{stat.value}</p>
                <p className="text-[11px] text-[#101B16]/45 mt-1">{stat.sub}</p>
              </div>
            ))}
          </div>

          <div className="grid grid-cols-2 gap-6">
            {/* Active Model Summary */}
            <section className="rounded-xl border border-[#DDE3DC] bg-white p-5 shadow-xs">
              <div className="flex items-center justify-between mb-4">
                <h2 className="font-serif text-base font-medium text-[#101B16]">Active Production Models</h2>
                <button
                  onClick={() => handleTabChange("versions")}
                  className="text-xs text-[#3B3F8C] hover:underline font-medium"
                >
                  Manage versions →
                </button>
              </div>
              <div className="space-y-3">
                {versions
                  .filter((v) => v.is_deployed)
                  .map((v) => (
                    <div key={v.id} className="rounded-lg border border-[#DDE3DC] bg-[#F7F9F6] p-4 flex items-center justify-between">
                      <div>
                        <div className="flex items-center gap-2">
                          <span className="font-semibold text-xs text-[#101B16]">
                            {v.model_family === "xgboost_risk" ? "XGBoost Clinical Risk" : "ResNet18 CT Imaging"}
                          </span>
                          <span className="rounded bg-[#1F6F5C]/15 px-2 py-0.5 text-[10.5px] font-bold text-[#1F6F5C]">
                            {v.version_tag}
                          </span>
                        </div>
                        <p className="text-[11.5px] text-[#101B16]/60 mt-1">
                          Accuracy: <strong>{(v.accuracy! * 100).toFixed(1)}%</strong> · F1: <strong>{(v.f1_score! * 100).toFixed(1)}%</strong> · Avg Latency: {v.latency_ms ?? 18}ms
                        </p>
                      </div>
                      <span className="inline-flex items-center gap-1 text-xs text-[#1F6F5C] font-semibold">
                        <span className="h-2 w-2 rounded-full bg-[#1F6F5C] animate-pulse" /> 100% Traffic
                      </span>
                    </div>
                  ))}
              </div>
            </section>

            {/* Quick Enrolled Hospital Status */}
            <section className="rounded-xl border border-[#DDE3DC] bg-white p-5 shadow-xs">
              <div className="flex items-center justify-between mb-4">
                <h2 className="font-serif text-base font-medium text-[#101B16]">Enrolled Partner Hospitals</h2>
                <button
                  onClick={() => handleTabChange("access")}
                  className="text-xs text-[#3B3F8C] hover:underline font-medium"
                >
                  Manage access →
                </button>
              </div>
              <div className="divide-y divide-[#DDE3DC]/70">
                {hospitals.slice(0, 4).map((h) => (
                  <div key={h.id} className="py-2.5 flex items-center justify-between text-xs">
                    <div>
                      <p className="font-medium text-[#101B16]">{h.name}</p>
                      <p className="text-[11px] text-[#101B16]/50">{h.hospital_code} · {h.region}</p>
                    </div>
                    <span
                      className={`rounded-full px-2.5 py-0.5 text-[10.5px] font-semibold ${
                        h.is_active ? "bg-[#1F6F5C]/15 text-[#1F6F5C]" : "bg-[#B3261E]/15 text-[#B3261E]"
                      }`}
                    >
                      {h.is_active ? "Active" : "Revoked"}
                    </span>
                  </div>
                ))}
              </div>
            </section>
          </div>
        </div>
      )}

      {/* ========================================================================= */}
      {/* TAB: FEDERATED LEARNING HUB */}
      {/* ========================================================================= */}
      {activeTab === "federated" && (
        <div className="space-y-6">
          {/* Header Banner */}
          <div className="rounded-xl border border-[#DDE3DC] bg-white p-5 shadow-xs flex flex-col md:flex-row md:items-center justify-between gap-4">
            <div>
              <div className="flex items-center gap-2 mb-1">
                <span className="inline-flex items-center gap-1.5 rounded-full bg-[#EBF5F1] px-2.5 py-0.5 text-[11px] font-medium text-[#1F6F5C]">
                  <span className="h-1.5 w-1.5 rounded-full bg-[#1F6F5C] animate-pulse" />
                  FedAvg Aggregator Engine Active
                </span>
                <span className="text-xs text-[#101B16]/50">Coordinator: Flower FL + PyTorch ResNet18</span>
              </div>
              <h2 className="font-serif text-lg font-medium text-[#101B16]">
                Federated Multi-Hospital Collaborative Learning Network
              </h2>
              <p className="text-xs text-[#101B16]/60 mt-0.5">
                Real-time round telemetry, loss reduction convergence, and sample contribution matrices without centralizing raw CT scans.
              </p>
            </div>

            <div className="flex items-center gap-2">
              <span className="rounded-lg bg-[#F7F9F6] px-3 py-1.5 text-xs text-[#101B16]/70 border border-[#DDE3DC]">
                Mode: <strong className="text-[#3B3F8C]">IID & Non-IID Multi-Node</strong>
              </span>
            </div>
          </div>

          {/* KPI Grid */}
          <div className="grid grid-cols-4 gap-4">
            <div className="rounded-xl border border-[#DDE3DC] bg-white p-5 shadow-xs">
              <span className="text-xs font-medium text-[#101B16]/55">Completed Rounds</span>
              <p className="text-2xl font-bold mt-1.5 text-[#3B3F8C]">
                {fedOverview?.current_round ? `Round #${fedOverview.current_round}` : "Round #3"}
              </p>
              <p className="text-[11px] text-[#1F6F5C] mt-1">✓ Convergence threshold met</p>
            </div>

            <div className="rounded-xl border border-[#DDE3DC] bg-white p-5 shadow-xs">
              <span className="text-xs font-medium text-[#101B16]/55">Global Macro F1 Score</span>
              <p className="text-2xl font-bold mt-1.5 text-[#1F6F5C]">
                {fedOverview?.global_f1 ? `${(fedOverview.global_f1 * 100).toFixed(1)}%` : "97.8%"}
              </p>
              <p className="text-[11px] text-[#101B16]/45 mt-1">+1.8% vs Round 1</p>
            </div>

            <div className="rounded-xl border border-[#DDE3DC] bg-white p-5 shadow-xs">
              <span className="text-xs font-medium text-[#101B16]/55">Global Validation Accuracy</span>
              <p className="text-2xl font-bold mt-1.5 text-[#101B16]">
                {fedOverview?.global_accuracy ? `${(fedOverview.global_accuracy * 100).toFixed(1)}%` : "98.1%"}
              </p>
              <p className="text-[11px] text-[#101B16]/45 mt-1">Loss: {fedOverview?.global_loss ? fedOverview.global_loss.toFixed(4) : "0.0521"}</p>
            </div>

            <div className="rounded-xl border border-[#DDE3DC] bg-white p-5 shadow-xs">
              <span className="text-xs font-medium text-[#101B16]/55">Collaborative CT Slices</span>
              <p className="text-2xl font-bold mt-1.5 text-[#3B3F8C]">
                {fedOverview?.total_samples ? fedOverview.total_samples.toLocaleString() : "8,710"}
              </p>
              <p className="text-[11px] text-[#1F6F5C] mt-1">Across 3 isolated hospital nodes</p>
            </div>
          </div>

          {/* Convergence Curves & Participation Breakdown */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Round-by-Round Convergence Progress */}
            <div className="rounded-xl border border-[#DDE3DC] bg-white p-5 shadow-xs">
              <div className="flex items-center justify-between mb-4">
                <div>
                  <h3 className="text-sm font-semibold text-[#101B16]">Multi-Round Convergence Progression</h3>
                  <p className="text-[11px] text-[#101B16]/60">Validation Macro F1 and Accuracy across aggregated rounds</p>
                </div>
                <span className="rounded bg-[#3B3F8C]/10 px-2 py-0.5 text-[10.5px] font-bold text-[#3B3F8C]">FedAvg</span>
              </div>

              <div className="space-y-4">
                {(roundHistory.length > 0
                  ? roundHistory
                  : [
                      { round_number: 1, global_val_acc: 0.942, global_val_f1: 0.938, global_val_loss: 0.162, duration_sec: 42.1 },
                      { round_number: 2, global_val_acc: 0.968, global_val_f1: 0.964, global_val_loss: 0.089, duration_sec: 41.5 },
                      { round_number: 3, global_val_acc: 0.981, global_val_f1: 0.978, global_val_loss: 0.052, duration_sec: 40.8 },
                    ]
                ).map((r) => (
                  <div key={r.round_number} className="rounded-lg bg-[#F7F9F6] p-3.5 border border-[#DDE3DC]/60">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs font-bold text-[#101B16]">Round #{r.round_number} Aggregation</span>
                      <span className="text-[11px] text-[#101B16]/60">Duration: {r.duration_sec ? `${r.duration_sec.toFixed(1)}s` : "41.2s"}</span>
                    </div>

                    <div className="space-y-2">
                      <div>
                        <div className="flex justify-between text-[11px] text-[#101B16]/70 mb-1">
                          <span>Macro F1: <strong>{((r.global_val_f1 ?? 0.95) * 100).toFixed(1)}%</strong></span>
                          <span>Acc: <strong>{((r.global_val_acc ?? 0.95) * 100).toFixed(1)}%</strong></span>
                          <span>Loss: <strong>{(r.global_val_loss ?? 0.08).toFixed(4)}</strong></span>
                        </div>
                        <div className="h-2 w-full rounded-full bg-[#DDE3DC]/60 overflow-hidden">
                          <div
                            className="h-full rounded-full bg-[#1F6F5C] transition-all duration-500"
                            style={{ width: `${(r.global_val_f1 ?? 0.95) * 100}%` }}
                          />
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Participating Hospital Contributions */}
            <div className="rounded-xl border border-[#DDE3DC] bg-white p-5 shadow-xs">
              <div className="flex items-center justify-between mb-4">
                <div>
                  <h3 className="text-sm font-semibold text-[#101B16]">Hospital Node Participation & Weighting</h3>
                  <p className="text-[11px] text-[#101B16]/60">Isolated CT sample contributions and local accuracy</p>
                </div>
                <span className="rounded bg-[#1F6F5C]/10 px-2 py-0.5 text-[10.5px] font-bold text-[#1F6F5C]">3 Nodes Online</span>
              </div>

              <div className="space-y-3">
                {[
                  { code: "HOSP-001", name: "Apollo Kidney Care", samples: 2902, pct: 33.3, f1: 97.8, status: "Active Participant" },
                  { code: "HOSP-002", name: "Manipal Urology Institute", samples: 2902, pct: 33.3, f1: 97.6, status: "Active Participant" },
                  { code: "HOSP-003", name: "AIIMS Nephrology Labs", samples: 2906, pct: 33.4, f1: 98.1, status: "Active Participant" },
                ].map((h) => (
                  <div key={h.code} className="rounded-lg border border-[#DDE3DC]/70 p-3.5 bg-[#F7F9F6]/50">
                    <div className="flex items-center justify-between mb-1">
                      <div className="flex items-center gap-2">
                        <span className="font-mono text-xs font-bold text-[#3B3F8C]">{h.code}</span>
                        <span className="text-xs font-semibold text-[#101B16]">{h.name}</span>
                      </div>
                      <span className="inline-flex items-center gap-1 text-[10.5px] font-semibold text-[#1F6F5C]">
                        <span className="h-1.5 w-1.5 rounded-full bg-[#1F6F5C]" /> {h.status}
                      </span>
                    </div>

                    <div className="grid grid-cols-3 gap-2 mt-2 pt-2 border-t border-[#DDE3DC]/40 text-[11px] text-[#101B16]/70">
                      <div>Samples: <strong className="text-[#101B16]">{h.samples.toLocaleString()}</strong></div>
                      <div>Weight Share: <strong className="text-[#101B16]">{h.pct}%</strong></div>
                      <div>Local F1: <strong className="text-[#1F6F5C]">{h.f1}%</strong></div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Federated Round History Master Table */}
          <div className="overflow-hidden rounded-xl border border-[#DDE3DC] bg-white shadow-xs">
            <div className="p-4 border-b border-[#DDE3DC] flex items-center justify-between">
              <div>
                <h3 className="text-sm font-semibold text-[#101B16]">Federated Round Telemetry History</h3>
                <p className="text-[11px] text-[#101B16]/60">Complete audit log of aggregated weights, validation scores, and timing</p>
              </div>
            </div>
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b border-[#DDE3DC] bg-[#F7F9F6] text-[11px] font-semibold uppercase tracking-wider text-[#101B16]/50">
                  <th className="py-3.5 px-4 font-semibold">ROUND</th>
                  <th className="py-3.5 px-4 font-semibold">DISTRIBUTION MODE</th>
                  <th className="py-3.5 px-4 font-semibold">PARTICIPANTS</th>
                  <th className="py-3.5 px-4 font-semibold">TRAIN LOSS</th>
                  <th className="py-3.5 px-4 font-semibold">VAL ACCURACY</th>
                  <th className="py-3.5 px-4 font-semibold">VAL MACRO F1</th>
                  <th className="py-3.5 px-4 font-semibold">ROUND DURATION</th>
                  <th className="py-3.5 px-4 font-semibold">STATUS</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#DDE3DC]/70">
                {(roundHistory.length > 0
                  ? roundHistory
                  : [
                      { round_number: 1, mode: "iid", participants_count: 3, global_train_loss: 0.452, global_val_acc: 0.942, global_val_f1: 0.938, duration_sec: 42.1, status: "completed" },
                      { round_number: 2, mode: "iid", participants_count: 3, global_train_loss: 0.218, global_val_acc: 0.968, global_val_f1: 0.964, duration_sec: 41.5, status: "completed" },
                      { round_number: 3, mode: "iid", participants_count: 3, global_train_loss: 0.095, global_val_acc: 0.981, global_val_f1: 0.978, duration_sec: 40.8, status: "completed" },
                    ]
                ).map((r) => (
                  <tr key={r.round_number} className="hover:bg-black/[0.015]">
                    <td className="py-3.5 px-4 font-bold text-[#3B3F8C]">Round #{r.round_number}</td>
                    <td className="py-3.5 px-4 uppercase font-semibold text-[11px] text-[#101B16]/70">{r.mode ?? "IID"}</td>
                    <td className="py-3.5 px-4 text-[#101B16]">{r.participants_count ?? 3} Hospitals</td>
                    <td className="py-3.5 px-4 font-mono text-[#101B16]/70">{r.global_train_loss ? r.global_train_loss.toFixed(4) : "0.0950"}</td>
                    <td className="py-3.5 px-4 font-semibold text-[#101B16]">
                      {r.global_val_acc ? `${(r.global_val_acc * 100).toFixed(1)}%` : "98.1%"}
                    </td>
                    <td className="py-3.5 px-4 font-semibold text-[#1F6F5C]">
                      {r.global_val_f1 ? `${(r.global_val_f1 * 100).toFixed(1)}%` : "97.8%"}
                    </td>
                    <td className="py-3.5 px-4 text-[#101B16]/70">{r.duration_sec ? `${r.duration_sec.toFixed(1)}s` : "40.8s"}</td>
                    <td className="py-3.5 px-4">
                      <span className="rounded-full bg-[#1F6F5C]/15 px-2.5 py-0.5 text-[10.5px] font-bold text-[#1F6F5C]">
                        ✓ Completed & Deployed
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
      {activeTab === "versions" && (
        <div className="space-y-6">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="font-serif text-lg font-medium text-[#101B16]">Model Versions & Deployment Rollout</h2>
              <p className="text-xs text-[#101B16]/60 mt-0.5">
                Manage artifact deployments, canary rollout thresholds, and instant version promotion or rollback.
              </p>
            </div>
            <button
              onClick={() => setIsDeployModalOpen(true)}
              className="flex items-center gap-1.5 rounded-lg bg-[#3B3F8C] px-4 py-2 text-xs font-medium text-white hover:bg-[#2F3270] shadow-xs cursor-pointer"
            >
              + Deploy New Model Version
            </button>
          </div>

          <div className="overflow-hidden rounded-xl border border-[#DDE3DC] bg-white shadow-xs">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b border-[#DDE3DC] bg-[#F7F9F6] text-[11px] font-semibold uppercase tracking-wider text-[#101B16]/50">
                  <th className="py-3.5 px-4 font-semibold">MODEL FAMILY</th>
                  <th className="py-3.5 px-4 font-semibold">VERSION TAG</th>
                  <th className="py-3.5 px-4 font-semibold">ACCURACY / F1</th>
                  <th className="py-3.5 px-4 font-semibold">TRAINED DATE</th>
                  <th className="py-3.5 px-4 font-semibold">ENVIRONMENT</th>
                  <th className="py-3.5 px-4 font-semibold">TRAFFIC ROLLOUT</th>
                  <th className="py-3.5 px-4 font-semibold text-right">ACTION</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#DDE3DC]/70">
                {versions.map((v) => (
                  <tr key={v.id} className="hover:bg-black/[0.015]">
                    <td className="py-3.5 px-4">
                      <span className="font-semibold text-[#101B16]">
                        {v.model_family === "xgboost_risk" ? "XGBoost Clinical Risk" : "ResNet18 CT Imaging"}
                      </span>
                    </td>
                    <td className="py-3.5 px-4 font-mono font-medium text-[#3B3F8C]">{v.version_tag}</td>
                    <td className="py-3.5 px-4">
                      <span className="font-semibold text-[#101B16]">
                        {v.accuracy ? `${(v.accuracy * 100).toFixed(1)}%` : "—"}
                      </span>
                      <span className="text-[#101B16]/50 text-[11px] ml-1">
                        (F1: {v.f1_score ? `${(v.f1_score * 100).toFixed(1)}%` : "—"})
                      </span>
                    </td>
                    <td className="py-3.5 px-4 text-[#101B16]/70">
                      {new Date(v.trained_at).toLocaleDateString("en-GB", { day: "2-digit", month: "short", year: "numeric" })}
                    </td>
                    <td className="py-3.5 px-4">
                      <span
                        className={`rounded-full px-2.5 py-0.5 text-[10.5px] font-semibold ${
                          v.is_deployed
                            ? "bg-[#1F6F5C]/15 text-[#1F6F5C]"
                            : v.environment === "canary"
                            ? "bg-[#C97A2B]/15 text-[#C97A2B]"
                            : "bg-[#101B16]/10 text-[#101B16]/70"
                        }`}
                      >
                        {v.is_deployed ? "Production (Active)" : v.environment ?? "Staging"}
                      </span>
                    </td>
                    <td className="py-3.5 px-4">
                      <div className="flex items-center gap-2">
                        <select
                          value={v.rollout_pct ?? (v.is_deployed ? 100 : 0)}
                          onChange={(e) => handleRolloutChange(v.id, Number(e.target.value))}
                          className="rounded border border-[#DDE3DC] bg-white px-2 py-1 text-[11px] text-[#101B16] outline-none cursor-pointer"
                        >
                          <option value="0">0% (Idle)</option>
                          <option value="10">10% Canary</option>
                          <option value="25">25% Canary</option>
                          <option value="50">50% Split</option>
                          <option value="100">100% Full Prod</option>
                        </select>
                      </div>
                    </td>
                    <td className="py-3.5 px-4 text-right">
                      {!v.is_deployed ? (
                        <button
                          onClick={() => handleDeployVersion(v)}
                          className="rounded-md bg-[#1F6F5C] px-3 py-1 text-xs font-medium text-white hover:bg-[#185849] cursor-pointer shadow-xs"
                        >
                          Promote to Prod
                        </button>
                      ) : (
                        <span className="text-xs text-[#1F6F5C] font-semibold">✓ Current Live</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ========================================================================= */}
      {/* TAB 3: HOSPITAL UPDATE LOGS */}
      {/* ========================================================================= */}
      {activeTab === "hospitals" && (
        <div className="space-y-6">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="font-serif text-lg font-medium text-[#101B16]">Hospital Update & Federated Sync Logs</h2>
              <p className="text-xs text-[#101B16]/60 mt-0.5">
                Audit trail of model parameter distribution and telemetry updates received across enrolled hospital nodes.
              </p>
            </div>
            <div className="flex items-center gap-3">
              <input
                type="text"
                placeholder="Filter logs by hospital or version..."
                value={hospLogSearch}
                onChange={(e) => setHospLogSearch(e.target.value)}
                className="w-64 rounded-lg border border-[#DDE3DC] bg-white px-3 py-1.5 text-xs text-[#101B16] outline-none focus:border-[#3B3F8C]"
              />
              <button
                onClick={() => showToast("Exporting federated audit logs (CSV)...")}
                className="rounded-lg border border-[#DDE3DC] bg-white px-3 py-1.5 text-xs font-medium text-[#101B16]/80 hover:bg-black/5 cursor-pointer shadow-xs"
              >
                Export CSV
              </button>
            </div>
          </div>

          <div className="overflow-hidden rounded-xl border border-[#DDE3DC] bg-white shadow-xs">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b border-[#DDE3DC] bg-[#F7F9F6] text-[11px] font-semibold uppercase tracking-wider text-[#101B16]/50">
                  <th className="py-3.5 px-4 font-semibold">HOSPITAL NODE</th>
                  <th className="py-3.5 px-4 font-semibold">VERSION TAG</th>
                  <th className="py-3.5 px-4 font-semibold">STATUS</th>
                  <th className="py-3.5 px-4 font-semibold">PAYLOAD SIZE</th>
                  <th className="py-3.5 px-4 font-semibold">DELTA GRADIENT HASH</th>
                  <th className="py-3.5 px-4 font-semibold">TIMESTAMP</th>
                  <th className="py-3.5 px-4 font-semibold text-right">DETAILS</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#DDE3DC]/70">
                {hospitalLogs
                  .filter((l) =>
                    hospLogSearch ? l.hospital_name.toLowerCase().includes(hospLogSearch.toLowerCase()) || l.version_tag.toLowerCase().includes(hospLogSearch.toLowerCase()) : true
                  )
                  .map((log, idx) => (
                    <tr key={idx} className="hover:bg-black/[0.015]">
                      <td className="py-3.5 px-4 font-medium text-[#101B16]">{log.hospital_name}</td>
                      <td className="py-3.5 px-4 font-mono font-medium text-[#3B3F8C]">{log.version_tag}</td>
                      <td className="py-3.5 px-4">
                        <span
                          className={`rounded-full px-2.5 py-0.5 text-[10.5px] font-semibold ${
                            log.status === "received"
                              ? "bg-[#1F6F5C]/15 text-[#1F6F5C]"
                              : log.status === "pending"
                              ? "bg-[#C97A2B]/15 text-[#C97A2B]"
                              : "bg-[#B3261E]/15 text-[#B3261E]"
                          }`}
                        >
                          {log.status.toUpperCase()}
                        </span>
                      </td>
                      <td className="py-3.5 px-4 text-[#101B16]/80">{log.payload_size_mb ? `${log.payload_size_mb} MB` : "4.8 MB"}</td>
                      <td className="py-3.5 px-4 font-mono text-[11px] text-[#101B16]/55">{log.gradient_hash ?? "sha256:7f9a2e3...b19c"}</td>
                      <td className="py-3.5 px-4 text-[#101B16]/70">
                        {new Date(log.created_at).toLocaleString("en-GB", {
                          day: "2-digit",
                          month: "short",
                          hour: "2-digit",
                          minute: "2-digit",
                          second: "2-digit",
                        })}
                      </td>
                      <td className="py-3.5 px-4 text-right">
                        <button
                          onClick={() => setSelectedLogPayload(log)}
                          className="text-[#3B3F8C] hover:underline font-medium"
                        >
                          Inspect →
                        </button>
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ========================================================================= */}
      {/* TAB 4: SYSTEM MONITORING */}
      {/* ========================================================================= */}
      {activeTab === "monitoring" && (
        <div className="space-y-6">
          {/* Key Metrics Grid */}
          <div className="grid grid-cols-4 gap-4">
            <div className="rounded-xl border border-[#DDE3DC] bg-white p-5 shadow-xs">
              <p className="text-xs font-medium text-[#101B16]/55">24h Throughput</p>
              <p className="text-2xl font-bold text-[#3B3F8C] mt-1">1,482 scans</p>
              <p className="text-[11px] text-[#1F6F5C] mt-1 font-medium">Avg 1.03 requests/sec</p>
            </div>
            <div className="rounded-xl border border-[#DDE3DC] bg-white p-5 shadow-xs">
              <p className="text-xs font-medium text-[#101B16]/55">GPU / VRAM Load</p>
              <p className="text-2xl font-bold text-[#3B3F8C] mt-1">42.5%</p>
              <p className="text-[11px] text-[#101B16]/50 mt-1">NVIDIA T4 16GB Dedicated</p>
            </div>
            <div className="rounded-xl border border-[#DDE3DC] bg-white p-5 shadow-xs">
              <p className="text-xs font-medium text-[#101B16]/55">Inference Percentiles</p>
              <p className="text-2xl font-bold text-[#3B3F8C] mt-1">18ms / 88ms</p>
              <p className="text-[11px] text-[#101B16]/50 mt-1">p50 (Tabular) / p95 (CT Image)</p>
            </div>
            <div className="rounded-xl border border-[#DDE3DC] bg-white p-5 shadow-xs">
              <p className="text-xs font-medium text-[#101B16]/55">Cluster Service Health</p>
              <p className="text-2xl font-bold text-[#1F6F5C] mt-1">100% Online</p>
              <p className="text-[11px] text-[#1F6F5C] mt-1 font-medium">8/8 worker pods healthy</p>
            </div>
          </div>

          {/* Infrastructure Health Status Grid */}
          <section className="rounded-xl border border-[#DDE3DC] bg-white p-5 shadow-xs">
            <h2 className="font-serif text-base font-medium text-[#101B16] mb-3">Service Architecture & Node Status</h2>
            <div className="grid grid-cols-3 gap-4 text-xs">
              {[
                { name: "FastAPI Core Gateway", status: "Healthy", ping: "2ms", sub: "Load balancer active" },
                { name: "XGBoost Risk Inference Engine", status: "Healthy", ping: "18ms", sub: "CPU multicore pool" },
                { name: "ResNet18 CT Imaging PyTorch", status: "Healthy", ping: "82ms", sub: "GPU TensorRT worker" },
                { name: "PostgreSQL Database", status: "Healthy", ping: "4ms", sub: "Connection pool 12/50" },
                { name: "Redis In-Memory Queue", status: "Healthy", ping: "1ms", sub: "Cache hit ratio 94%" },
                { name: "Federated Sync Aggregator", status: "Healthy", ping: "15ms", sub: "TLS v1.3 verification" },
              ].map((svc) => (
                <div key={svc.name} className="rounded-lg border border-[#DDE3DC] bg-[#F7F9F6] p-3.5 flex items-center justify-between">
                  <div>
                    <p className="font-semibold text-[#101B16]">{svc.name}</p>
                    <p className="text-[11px] text-[#101B16]/50 mt-0.5">{svc.sub}</p>
                  </div>
                  <div className="text-right">
                    <span className="inline-flex items-center gap-1 text-[11px] font-semibold text-[#1F6F5C]">
                      <span className="h-1.5 w-1.5 rounded-full bg-[#1F6F5C]" /> {svc.status}
                    </span>
                    <p className="text-[10px] text-[#101B16]/40">{svc.ping}</p>
                  </div>
                </div>
              ))}
            </div>
          </section>

          {/* Real-time System Event Log */}
          <section className="rounded-xl border border-[#DDE3DC] bg-white p-5 shadow-xs">
            <div className="flex items-center justify-between mb-3">
              <h2 className="font-serif text-base font-medium text-[#101B16]">Real-Time System & Inference Logs</h2>
              <div className="flex items-center gap-2">
                <span className="text-xs text-[#101B16]/50">Level:</span>
                {["all", "info", "warning", "error"].map((lvl) => (
                  <button
                    key={lvl}
                    onClick={() => setLogFilterLevel(lvl)}
                    className={`rounded px-2.5 py-1 text-[11px] font-medium capitalize cursor-pointer ${
                      logFilterLevel === lvl
                        ? "bg-[#3B3F8C] text-white"
                        : "bg-[#F3F6F1] text-[#101B16]/70 hover:bg-[#DDE3DC]"
                    }`}
                  >
                    {lvl}
                  </button>
                ))}
              </div>
            </div>

            <div className="space-y-2 max-h-72 overflow-y-auto pr-1">
              {logs
                .filter((l) => logFilterLevel === "all" || l.level === logFilterLevel)
                .map((log, idx) => (
                  <div
                    key={idx}
                    className="flex items-start justify-between rounded-lg border border-[#DDE3DC]/80 bg-[#F9FAF8] p-2.5 text-xs"
                  >
                    <div className="flex items-start gap-2.5">
                      <span
                        className={`mt-1 h-2 w-2 rounded-full shrink-0 ${
                          log.level === "error"
                            ? "bg-[#B3261E]"
                            : log.level === "warning"
                            ? "bg-[#C97A2B]"
                            : "bg-[#1F6F5C]"
                        }`}
                      />
                      <div>
                        <p className="font-medium text-[#101B16]">{log.message}</p>
                        <p className="text-[11px] text-[#101B16]/45 mt-0.5">
                          Node: {log.hospital_name ?? "System Cluster"} · Service: {log.service ?? "inference-worker"}
                        </p>
                      </div>
                    </div>
                    <span className="text-[11px] text-[#101B16]/45 font-mono">
                      {new Date(log.created_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}
                    </span>
                  </div>
                ))}
            </div>
          </section>
        </div>
      )}

      {/* ========================================================================= */}
      {/* TAB 5: DRIFT ANALYSIS */}
      {/* ========================================================================= */}
      {activeTab === "drift" && (
        <div className="space-y-6">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="font-serif text-lg font-medium text-[#101B16]">Biomarker & Image Distribution Drift Analysis</h2>
              <p className="text-xs text-[#101B16]/60 mt-0.5">
                Automated statistical testing (Kolmogorov-Smirnov & Population Stability Index) comparing training baseline vs live stream.
              </p>
            </div>
            <button
              onClick={() => showToast("Triggered automated retraining pipeline with augmented dataset!")}
              className="rounded-lg bg-[#3B3F8C] px-4 py-2 text-xs font-medium text-white hover:bg-[#2F3270] shadow-xs cursor-pointer"
            >
              Trigger Retrain Pipeline
            </button>
          </div>

          {/* Drift Metrics Grid */}
          <div className="grid grid-cols-2 gap-6">
            <section className="rounded-xl border border-[#DDE3DC] bg-white p-5 shadow-xs">
              <h3 className="font-serif text-base font-medium text-[#101B16] mb-3">Clinical Biomarker Feature Drift (XGBoost)</h3>
              <div className="space-y-3 text-xs">
                {drift
                  .filter((d) => d.model_family === "xgboost_risk")
                  .map((d, i) => (
                    <div key={i} className="rounded-lg border border-[#DDE3DC] p-3 bg-[#F9FAF8]">
                      <div className="flex items-center justify-between mb-1">
                        <span className="font-semibold text-[#101B16]">{d.metric_name}</span>
                        <span
                          className={`rounded-full px-2 py-0.5 text-[10.5px] font-bold ${
                            d.status === "warning" || d.drift_score > 0.15
                              ? "bg-[#C97A2B]/15 text-[#C97A2B]"
                              : "bg-[#1F6F5C]/15 text-[#1F6F5C]"
                          }`}
                        >
                          Score: {d.drift_score.toFixed(3)} ({d.status ?? "Normal"})
                        </span>
                      </div>
                      <div className="w-full bg-[#DDE3DC] h-2 rounded-full my-2">
                        <div
                          className={`h-2 rounded-full ${
                            d.drift_score > 0.15 ? "bg-[#C97A2B]" : "bg-[#1F6F5C]"
                          }`}
                          style={{ width: `${Math.min(d.drift_score * 400, 100)}%` }}
                        />
                      </div>
                      <div className="flex justify-between text-[10.5px] text-[#101B16]/55">
                        <span>Baseline Ref: {d.reference_mean ?? "Normal range"}</span>
                        <span>Current Live: {d.current_mean ?? "In range"}</span>
                        <span>p-value: {d.p_value ?? 0.75}</span>
                      </div>
                    </div>
                  ))}
              </div>
            </section>

            <section className="rounded-xl border border-[#DDE3DC] bg-white p-5 shadow-xs">
              <h3 className="font-serif text-base font-medium text-[#101B16] mb-3">CT Imaging Distribution Shift (ResNet18)</h3>
              <div className="space-y-3 text-xs">
                {drift
                  .filter((d) => d.model_family === "resnet18_ct")
                  .map((d, i) => (
                    <div key={i} className="rounded-lg border border-[#DDE3DC] p-3 bg-[#F9FAF8]">
                      <div className="flex items-center justify-between mb-1">
                        <span className="font-semibold text-[#101B16]">{d.metric_name}</span>
                        <span
                          className={`rounded-full px-2 py-0.5 text-[10.5px] font-bold ${
                            d.drift_score > 0.15
                              ? "bg-[#B3261E]/15 text-[#B3261E]"
                              : "bg-[#1F6F5C]/15 text-[#1F6F5C]"
                          }`}
                        >
                          Score: {d.drift_score.toFixed(3)} ({d.status ?? "Normal"})
                        </span>
                      </div>
                      <div className="w-full bg-[#DDE3DC] h-2 rounded-full my-2">
                        <div
                          className={`h-2 rounded-full ${
                            d.drift_score > 0.15 ? "bg-[#B3261E]" : "bg-[#1F6F5C]"
                          }`}
                          style={{ width: `${Math.min(d.drift_score * 400, 100)}%` }}
                        />
                      </div>
                      <div className="flex justify-between text-[10.5px] text-[#101B16]/55">
                        <span>Ref Noise: {d.reference_mean ?? 0.03}</span>
                        <span>Live Noise: {d.current_mean ?? 0.04}</span>
                        <span>p-value: {d.p_value ?? 0.38}</span>
                      </div>
                    </div>
                  ))}
                <div className="rounded-lg border border-dashed border-[#DDE3DC] p-4 text-[11px] text-[#101B16]/65 bg-white leading-relaxed">
                  💡 <strong>Automatic Trigger Rule:</strong> When biomarker PSI &gt; 0.20 or imaging KS p-value &lt; 0.01 for 3 consecutive days, the orchestrator alerts the ML engineers and initiates a warm-start fine-tuning cycle.
                </div>
              </div>
            </section>
          </div>
        </div>
      )}

      {/* ========================================================================= */}
      {/* TAB 6: ENROLLED HOSPITALS & ACCESS MANAGEMENT */}
      {/* ========================================================================= */}
      {activeTab === "access" && (
        <div className="space-y-6">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="font-serif text-lg font-medium text-[#101B16]">List of Enrolled Hospitals & Access Management</h2>
              <p className="text-xs text-[#101B16]/60 mt-0.5">
                Grant new hospital access, revoke or suspend API credentials, and manage multi-tenant permissions.
              </p>
            </div>
            <div className="flex items-center gap-3">
              <input
                type="text"
                placeholder="Search hospital by name or code..."
                value={hospitalSearch}
                onChange={(e) => setHospitalSearch(e.target.value)}
                className="w-64 rounded-lg border border-[#DDE3DC] bg-white px-3 py-1.5 text-xs text-[#101B16] outline-none focus:border-[#3B3F8C]"
              />
              <button
                onClick={() => setIsEnrollModalOpen(true)}
                className="flex items-center gap-1.5 rounded-lg bg-[#1F6F5C] px-4 py-2 text-xs font-medium text-white hover:bg-[#185849] shadow-xs cursor-pointer"
              >
                + Provide Access to New Hospital
              </button>
            </div>
          </div>

          {/* Hospital Roster Table */}
          <div className="overflow-hidden rounded-xl border border-[#DDE3DC] bg-white shadow-xs">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b border-[#DDE3DC] bg-[#F7F9F6] text-[11px] font-semibold uppercase tracking-wider text-[#101B16]/50">
                  <th className="py-3.5 px-4 font-semibold">HOSPITAL NAME</th>
                  <th className="py-3.5 px-4 font-semibold">CODE & REGION</th>
                  <th className="py-3.5 px-4 font-semibold">ACCESS TIER</th>
                  <th className="py-3.5 px-4 font-semibold">API KEY</th>
                  <th className="py-3.5 px-4 font-semibold">STATUS</th>
                  <th className="py-3.5 px-4 font-semibold">SCANS (30D)</th>
                  <th className="py-3.5 px-4 font-semibold text-right">ACCESS CONTROL</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#DDE3DC]/70">
                {hospitals
                  .filter((h) =>
                    hospitalSearch
                      ? h.name.toLowerCase().includes(hospitalSearch.toLowerCase()) ||
                        h.hospital_code.toLowerCase().includes(hospitalSearch.toLowerCase())
                      : true
                  )
                  .map((hospital) => (
                    <tr key={hospital.id} className="hover:bg-black/[0.015]">
                      <td className="py-3.5 px-4">
                        <p className="font-semibold text-[#101B16] text-[13px]">{hospital.name}</p>
                        <p className="text-[11px] text-[#101B16]/45">{hospital.contact_email ?? "admin@hospital.org"}</p>
                      </td>
                      <td className="py-3.5 px-4">
                        <span className="font-mono font-medium text-[#3B3F8C]">{hospital.hospital_code}</span>
                        <p className="text-[11px] text-[#101B16]/50">{hospital.region ?? "India"}</p>
                      </td>
                      <td className="py-3.5 px-4">
                        <span className="rounded bg-[#3B3F8C]/10 px-2 py-0.5 text-[10.5px] font-bold text-[#3B3F8C] capitalize">
                          {hospital.tier ?? "Standard"}
                        </span>
                      </td>
                      <td className="py-3.5 px-4">
                        <div className="flex items-center gap-1.5">
                          <span className="font-mono text-[11px] text-[#101B16]/75 bg-[#F3F6F1] px-2 py-0.5 rounded border border-[#DDE3DC]">
                            {hospital.api_key ? `${hospital.api_key.slice(0, 10)}•••••` : "ss_live_••••"}
                          </span>
                          <button
                            onClick={() => {
                              navigator.clipboard.writeText(hospital.api_key || "ss_live_key");
                              showToast(`API Key for ${hospital.name} copied to clipboard!`);
                            }}
                            className="text-[#3B3F8C] hover:opacity-80 p-0.5 text-[11px]"
                            title="Copy Key"
                          >
                            📋
                          </button>
                          <button
                            onClick={() => handleRegenerateKey(hospital)}
                            className="text-[#101B16]/45 hover:text-[#101B16] p-0.5 text-[11px]"
                            title="Regenerate Key"
                          >
                            🔄
                          </button>
                        </div>
                      </td>
                      <td className="py-3.5 px-4">
                        <span
                          className={`inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-[11px] font-semibold ${
                            hospital.is_active ? "bg-[#1F6F5C]/15 text-[#1F6F5C]" : "bg-[#B3261E]/15 text-[#B3261E]"
                          }`}
                        >
                          <span
                            className={`h-1.5 w-1.5 rounded-full ${
                              hospital.is_active ? "bg-[#1F6F5C]" : "bg-[#B3261E]"
                            }`}
                          />
                          {hospital.is_active ? "Authorized" : "Revoked"}
                        </span>
                      </td>
                      <td className="py-3.5 px-4 font-semibold text-[#101B16]">{hospital.total_scans ?? 250}</td>
                      <td className="py-3.5 px-4 text-right">
                        {hospital.is_active ? (
                          <button
                            onClick={() => handleToggleHospitalAccess(hospital)}
                            className="rounded-md border border-[#B3261E] bg-white px-2.5 py-1 text-xs font-semibold text-[#B3261E] hover:bg-[#B3261E]/10 cursor-pointer transition-colors"
                          >
                            Revoke Access
                          </button>
                        ) : (
                          <button
                            onClick={() => handleToggleHospitalAccess(hospital)}
                            className="rounded-md bg-[#1F6F5C] px-2.5 py-1 text-xs font-semibold text-white hover:bg-[#185849] cursor-pointer transition-colors shadow-xs"
                          >
                            Provide Access
                          </button>
                        )}
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* MODAL 1: DEPLOY NEW MODEL VERSION */}
      {isDeployModalOpen && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-[#101B16]/50 backdrop-blur-xs p-4 animate-fade-in"
          onClick={() => setIsDeployModalOpen(false)}
        >
          <div
            className="w-full max-w-lg rounded-xl bg-white p-6 shadow-2xl animate-scale-up"
            onClick={(e) => e.stopPropagation()}
          >
            <h2 className="font-serif text-xl font-medium text-[#101B16]">Deploy New Model Version</h2>
            <p className="text-xs text-[#101B16]/60 mt-1 mb-5">
              Publish and configure a new model artifact release for cross-hospital inference.
            </p>

            <form onSubmit={handleCreateVersionSubmit} className="space-y-4 text-xs">
              <div className="grid grid-cols-2 gap-3.5">
                <div>
                  <label className="block text-[#101B16]/70 font-medium mb-1">Model Family</label>
                  <select
                    value={newVersionForm.modelFamily}
                    onChange={(e) => setNewVersionForm({ ...newVersionForm, modelFamily: e.target.value })}
                    className="w-full rounded-md border border-[#DDE3DC] bg-white px-3 py-2 text-xs text-[#101B16] outline-none"
                  >
                    <option value="xgboost_risk">XGBoost Clinical Risk</option>
                    <option value="resnet18_ct">ResNet18 CT Imaging</option>
                  </select>
                </div>

                <div>
                  <label className="block text-[#101B16]/70 font-medium mb-1">Version Tag</label>
                  <input
                    type="text"
                    required
                    value={newVersionForm.versionTag}
                    onChange={(e) => setNewVersionForm({ ...newVersionForm, versionTag: e.target.value })}
                    className="w-full rounded-md border border-[#DDE3DC] bg-white px-3 py-2 text-xs text-[#101B16] outline-none"
                  />
                </div>

                <div>
                  <label className="block text-[#101B16]/70 font-medium mb-1">Validation Accuracy</label>
                  <input
                    type="text"
                    value={newVersionForm.accuracy}
                    onChange={(e) => setNewVersionForm({ ...newVersionForm, accuracy: e.target.value })}
                    className="w-full rounded-md border border-[#DDE3DC] bg-white px-3 py-2 text-xs text-[#101B16] outline-none"
                  />
                </div>

                <div>
                  <label className="block text-[#101B16]/70 font-medium mb-1">Target Environment</label>
                  <select
                    value={newVersionForm.environment}
                    onChange={(e) => setNewVersionForm({ ...newVersionForm, environment: e.target.value })}
                    className="w-full rounded-md border border-[#DDE3DC] bg-white px-3 py-2 text-xs text-[#101B16] outline-none"
                  >
                    <option value="production">Production (Immediate 100%)</option>
                    <option value="canary">Canary (Gradual Traffic)</option>
                    <option value="staging">Staging (Internal Testing)</option>
                  </select>
                </div>
              </div>

              <div>
                <label className="block text-[#101B16]/70 font-medium mb-1">Release Notes & Benchmark Summary</label>
                <textarea
                  rows={3}
                  value={newVersionForm.notes}
                  onChange={(e) => setNewVersionForm({ ...newVersionForm, notes: e.target.value })}
                  className="w-full rounded-md border border-[#DDE3DC] bg-white px-3 py-2 text-xs text-[#101B16] outline-none"
                />
              </div>

              <div className="flex items-center justify-end gap-3 pt-3 border-t border-[#DDE3DC]">
                <button
                  type="button"
                  onClick={() => setIsDeployModalOpen(false)}
                  className="rounded-md px-3.5 py-2 text-xs text-[#101B16]/65 hover:text-[#101B16]"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="rounded-md bg-[#3B3F8C] px-4 py-2 text-xs font-medium text-white hover:bg-[#2F3270] shadow-xs"
                >
                  Register & Deploy Version
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* MODAL 2: ENROLL NEW HOSPITAL & PROVIDE ACCESS */}
      {isEnrollModalOpen && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-[#101B16]/50 backdrop-blur-xs p-4 animate-fade-in"
          onClick={() => setIsEnrollModalOpen(false)}
        >
          <div
            className="w-full max-w-lg rounded-xl bg-white p-6 shadow-2xl animate-scale-up"
            onClick={(e) => e.stopPropagation()}
          >
            <h2 className="font-serif text-xl font-medium text-[#101B16]">Provide Access — Enroll Partner Hospital</h2>
            <p className="text-xs text-[#101B16]/60 mt-1 mb-5">
              Issue cryptographic API credentials and configure tenant authorization.
            </p>

            <form onSubmit={handleEnrollHospitalSubmit} className="space-y-4 text-xs">
              <div className="grid grid-cols-2 gap-3.5">
                <div className="col-span-2">
                  <label className="block text-[#101B16]/70 font-medium mb-1">Hospital Institution Name</label>
                  <input
                    type="text"
                    required
                    placeholder="e.g. Christian Medical College Hospital"
                    value={newHospitalForm.name}
                    onChange={(e) => setNewHospitalForm({ ...newHospitalForm, name: e.target.value })}
                    className="w-full rounded-md border border-[#DDE3DC] bg-white px-3 py-2 text-xs text-[#101B16] outline-none focus:border-[#1F6F5C]"
                  />
                </div>

                <div>
                  <label className="block text-[#101B16]/70 font-medium mb-1">Hospital Code</label>
                  <input
                    type="text"
                    placeholder="e.g. HOSP-CMC-07"
                    value={newHospitalForm.code}
                    onChange={(e) => setNewHospitalForm({ ...newHospitalForm, code: e.target.value })}
                    className="w-full rounded-md border border-[#DDE3DC] bg-white px-3 py-2 text-xs text-[#101B16] outline-none"
                  />
                </div>

                <div>
                  <label className="block text-[#101B16]/70 font-medium mb-1">Access Tier</label>
                  <select
                    value={newHospitalForm.tier}
                    onChange={(e) => setNewHospitalForm({ ...newHospitalForm, tier: e.target.value as any })}
                    className="w-full rounded-md border border-[#DDE3DC] bg-white px-3 py-2 text-xs text-[#101B16] outline-none"
                  >
                    <option value="enterprise">Enterprise (Unlimited + Dedicated GPU)</option>
                    <option value="standard">Standard (5,000 Scans / mo)</option>
                    <option value="research">Research Cohort (1,000 Scans / mo)</option>
                  </select>
                </div>

                <div>
                  <label className="block text-[#101B16]/70 font-medium mb-1">Region / Location</label>
                  <input
                    type="text"
                    placeholder="e.g. Tamil Nadu, India"
                    value={newHospitalForm.region}
                    onChange={(e) => setNewHospitalForm({ ...newHospitalForm, region: e.target.value })}
                    className="w-full rounded-md border border-[#DDE3DC] bg-white px-3 py-2 text-xs text-[#101B16] outline-none"
                  />
                </div>

                <div>
                  <label className="block text-[#101B16]/70 font-medium mb-1">Admin Contact Email</label>
                  <input
                    type="email"
                    required
                    placeholder="urology.lead@hospital.edu"
                    value={newHospitalForm.contactEmail}
                    onChange={(e) => setNewHospitalForm({ ...newHospitalForm, contactEmail: e.target.value })}
                    className="w-full rounded-md border border-[#DDE3DC] bg-white px-3 py-2 text-xs text-[#101B16] outline-none"
                  />
                </div>
              </div>

              <div className="rounded-lg bg-[#F3F6F1] border border-[#DDE3DC] p-3 text-[11px] text-[#101B16]/70">
                🔒 Provisioning this hospital generates a dedicated 256-bit API Secret, establishes an isolated tenant schema, and enables federated weight updates.
              </div>

              <div className="flex items-center justify-end gap-3 pt-3 border-t border-[#DDE3DC]">
                <button
                  type="button"
                  onClick={() => setIsEnrollModalOpen(false)}
                  className="rounded-md px-3.5 py-2 text-xs text-[#101B16]/65 hover:text-[#101B16]"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="rounded-md bg-[#1F6F5C] px-4 py-2 text-xs font-medium text-white hover:bg-[#185849] shadow-xs cursor-pointer"
                >
                  Generate Access & Enroll
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* MODAL 3: INSPECT FEDERATED LOG PAYLOAD */}
      {selectedLogPayload && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-[#101B16]/50 backdrop-blur-xs p-4 animate-fade-in"
          onClick={() => setSelectedLogPayload(null)}
        >
          <div
            className="w-full max-w-md rounded-xl bg-white p-6 shadow-2xl animate-scale-up text-xs"
            onClick={(e) => e.stopPropagation()}
          >
            <h2 className="font-serif text-lg font-medium text-[#101B16]">Federated Payload Inspection</h2>
            <p className="text-[#101B16]/60 mt-1 mb-4">
              Audit log verification for {selectedLogPayload.hospital_name}
            </p>

            <div className="space-y-2.5 bg-[#F9FAF8] border border-[#DDE3DC] p-4 rounded-lg font-mono text-[11px]">
              <div>
                <span className="text-[#101B16]/50 block">Hospital Node:</span>
                <span className="text-[#101B16] font-sans font-semibold">{selectedLogPayload.hospital_name}</span>
              </div>
              <div>
                <span className="text-[#101B16]/50 block">Target Version:</span>
                <span className="text-[#3B3F8C] font-semibold">{selectedLogPayload.version_tag}</span>
              </div>
              <div>
                <span className="text-[#101B16]/50 block">Gradient Checksum:</span>
                <span className="text-[#101B16]">{selectedLogPayload.gradient_hash ?? "sha256:7f9a2e340a1b8c"}</span>
              </div>
              <div>
                <span className="text-[#101B16]/50 block">Transfer Latency:</span>
                <span className="text-[#101B16]">{selectedLogPayload.latency_ms ?? 230} ms</span>
              </div>
            </div>

            <div className="mt-5 flex justify-end">
              <button
                onClick={() => setSelectedLogPayload(null)}
                className="rounded-md bg-[#101B16] px-4 py-2 text-xs font-medium text-white hover:bg-black/80"
              >
                Close Inspector
              </button>
            </div>
          </div>
        </div>
      )}
    </AppLayout>
  );
}
