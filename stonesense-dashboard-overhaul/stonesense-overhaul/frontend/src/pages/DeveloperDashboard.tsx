// Drop at: frontend/src/pages/DeveloperDashboard.tsx
import { useEffect, useState } from "react";
import AppLayout from "../components/layout/AppLayout";
import {
  fetchModelPerformance,
  fetchHospitalLogs,
  fetchSystemMonitoring,
  fetchSystemLogs,
  fetchDriftAnalysis,
  deployModelVersion,
} from "../services/developerApi";
import {
  ModelPerformance,
  HospitalUpdateLogEntry,
  SystemLogEntry,
  DriftPoint,
  SystemMonitoringSummary,
} from "../types/dashboard";

export default function DeveloperDashboard() {
  const [versions, setVersions] = useState<ModelPerformance[]>([]);
  const [hospitalLogs, setHospitalLogs] = useState<HospitalUpdateLogEntry[]>([]);
  const [monitoring, setMonitoring] = useState<SystemMonitoringSummary | null>(null);
  const [logs, setLogs] = useState<SystemLogEntry[]>([]);
  const [drift, setDrift] = useState<DriftPoint[]>([]);

  const reload = () => {
    fetchModelPerformance().then(setVersions).catch(() => setVersions([]));
    fetchHospitalLogs().then(setHospitalLogs).catch(() => setHospitalLogs([]));
    fetchSystemMonitoring().then(setMonitoring).catch(() => setMonitoring(null));
    fetchSystemLogs().then(setLogs).catch(() => setLogs([]));
    fetchDriftAnalysis().then(setDrift).catch(() => setDrift([]));
  };

  useEffect(reload, []);

  const handleDeploy = async (id: number) => {
    await deployModelVersion(id);
    reload();
  };

  return (
    <AppLayout
      role="developer"
      title="Developer console"
      subtitle="Aggregate model & system health across all hospitals — no patient-level data."
    >
      <div className="grid grid-cols-4 gap-4 mb-6">
        {[
          { label: "Predictions (24h)", value: monitoring?.total_predictions_24h ?? "—" },
          { label: "Errors (24h)", value: monitoring?.error_count_24h ?? "—" },
          {
            label: "Avg latency",
            value: monitoring?.avg_latency_ms ? `${monitoring.avg_latency_ms} ms` : "—",
          },
          { label: "Uptime", value: monitoring ? `${monitoring.uptime_pct}%` : "—" },
        ].map((stat) => (
          <div key={stat.label} className="rounded-lg border border-stonesense-line bg-white p-4">
            <p className="text-xs text-stonesense-ink/50">{stat.label}</p>
            <p className="text-2xl font-semibold mt-1 text-stonesense-indigo">{stat.value}</p>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-2 gap-6">
        <section className="rounded-lg border border-stonesense-line bg-white p-5">
          <h2 className="font-serif text-base mb-3">Model performance & deployment</h2>
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-stonesense-ink/50 border-b border-stonesense-line">
                <th className="pb-2 font-normal">Model</th>
                <th className="pb-2 font-normal">Version</th>
                <th className="pb-2 font-normal">Accuracy</th>
                <th className="pb-2 font-normal">Status</th>
                <th className="pb-2 font-normal" />
              </tr>
            </thead>
            <tbody>
              {versions.map((v) => (
                <tr key={`${v.model_family}-${v.version_tag}`} className="border-b border-stonesense-line/60">
                  <td className="py-2">{v.model_family}</td>
                  <td className="py-2">{v.version_tag}</td>
                  <td className="py-2">{v.accuracy ? `${(v.accuracy * 100).toFixed(1)}%` : "—"}</td>
                  <td className="py-2">
                    {v.is_deployed ? (
                      <span className="text-stonesense-teal">Deployed</span>
                    ) : (
                      <span className="text-stonesense-ink/40">Idle</span>
                    )}
                  </td>
                  <td className="py-2 text-right">
                    {!v.is_deployed && (
                      <button
                        className="text-xs text-stonesense-indigo underline"
                        onClick={() => handleDeploy((v as any).id ?? 0)}
                      >
                        Deploy
                      </button>
                    )}
                  </td>
                </tr>
              ))}
              {versions.length === 0 && (
                <tr>
                  <td colSpan={5} className="py-4 text-center text-stonesense-ink/40">
                    No model versions recorded yet.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </section>

        <section className="rounded-lg border border-stonesense-line bg-white p-5">
          <h2 className="font-serif text-base mb-3">Federated update history</h2>
          <ul className="flex flex-col gap-2 text-sm max-h-64 overflow-y-auto">
            {hospitalLogs.map((log, i) => (
              <li key={i} className="flex justify-between border-b border-stonesense-line/60 pb-2">
                <span>{log.hospital_name}</span>
                <span className="text-stonesense-ink/50">{log.version_tag}</span>
                <span
                  className={
                    log.status === "received" ? "text-stonesense-teal" : "text-stonesense-amber"
                  }
                >
                  {log.status}
                </span>
              </li>
            ))}
            {hospitalLogs.length === 0 && (
              <li className="text-stonesense-ink/40 text-center py-4">No update history yet.</li>
            )}
          </ul>
        </section>

        <section className="rounded-lg border border-stonesense-line bg-white p-5">
          <h2 className="font-serif text-base mb-3">System monitoring — recent events</h2>
          <ul className="flex flex-col gap-2 text-sm max-h-64 overflow-y-auto">
            {logs.map((log, i) => (
              <li key={i} className="flex items-start gap-2 border-b border-stonesense-line/60 pb-2">
                <span
                  className={`mt-0.5 h-2 w-2 rounded-full shrink-0 ${
                    log.level === "error"
                      ? "bg-red-500"
                      : log.level === "warning"
                      ? "bg-stonesense-amber"
                      : "bg-stonesense-teal"
                  }`}
                />
                <span>
                  {log.message}
                  <span className="text-stonesense-ink/40"> — {log.hospital_name ?? "system"}</span>
                </span>
              </li>
            ))}
            {logs.length === 0 && (
              <li className="text-stonesense-ink/40 text-center py-4">No events logged yet.</li>
            )}
          </ul>
        </section>

        <section className="rounded-lg border border-stonesense-line bg-white p-5">
          <h2 className="font-serif text-base mb-3">Data / model drift</h2>
          <div className="flex flex-col gap-2 text-sm">
            {["xgboost_risk", "resnet18_ct"].map((family) => {
              const points = drift.filter((d) => d.model_family === family);
              const latest = points[points.length - 1];
              return (
                <div key={family} className="flex items-center justify-between border-b border-stonesense-line/60 pb-2">
                  <span>{family}</span>
                  <span className="text-stonesense-ink/50">{latest?.metric_name ?? "—"}</span>
                  <span
                    className={
                      latest && latest.drift_score > 0.15 ? "text-red-500" : "text-stonesense-teal"
                    }
                  >
                    {latest ? latest.drift_score.toFixed(3) : "—"}
                  </span>
                </div>
              );
            })}
            {drift.length === 0 && (
              <p className="text-stonesense-ink/40 text-center py-4">No drift snapshots yet.</p>
            )}
          </div>
        </section>
      </div>
    </AppLayout>
  );
}
