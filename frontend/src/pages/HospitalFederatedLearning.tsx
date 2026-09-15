import { useEffect, useMemo, useState } from "react";
import AppLayout from "../components/layout/AppLayout";
import { useHospital } from "../context/HospitalContext";
import {
  createFederatedWebSocket,
  fetchFederatedOverview,
  fetchHospitalCurrentModel,
  fetchHospitalDatasetStatus,
  fetchHospitalLiveStatus,
  fetchHospitalTrainingHistory,
} from "../services/federatedApi";
import {
  DatasetStatus,
  FederatedOverview,
  HospitalFLStatus,
  HospitalLiveStatus,
  HospitalRunTelemetry,
  FederatedEventMessage,
} from "../types/federated";

const lifecycle: Array<{ status: HospitalFLStatus; label: string }> = [
  { status: "MODEL_RECEIVED", label: "Global model received" },
  { status: "TRAINING", label: "Local training" },
  { status: "UPDATE_SUBMITTED", label: "Model update submitted" },
  { status: "WAITING_FOR_AGGREGATION", label: "Waiting for FedAvg" },
  { status: "MODEL_UPDATED", label: "Global model updated" },
  { status: "COMPLETED", label: "Round completed" },
];

const activeStatuses = new Set([
  "WAITING",
  "MODEL_RECEIVED",
  "TRAINING",
  "TRAINING_COMPLETED",
  "UPDATE_SUBMITTED",
  "WAITING_FOR_AGGREGATION",
  "MODEL_UPDATED",
]);

function formatMetric(value: number | null | undefined, percent = true) {
  if (value === null || value === undefined) return "-";
  return percent ? `${(value * 100).toFixed(1)}%` : value.toFixed(4);
}

function statusIndex(status: HospitalFLStatus) {
  if (status === "WAITING") return -1;
  if (status === "TRAINING_COMPLETED") return 1;
  if (status === "FAILED") return -2;
  return lifecycle.findIndex((item) => item.status === status);
}

export default function HospitalFederatedLearning() {
  const { hospitals, hospitalId } = useHospital();
  const activeHospitalId = hospitalId ?? 1;
  const hospital = hospitals.find((item) => item.id === activeHospitalId);
  const hospitalCode = hospital?.hospital_code;

  const [live, setLive] = useState<HospitalLiveStatus | null>(null);
  const [overview, setOverview] = useState<FederatedOverview | null>(null);
  const [currentModel, setCurrentModel] = useState<Awaited<ReturnType<typeof fetchHospitalCurrentModel>> | null>(null);
  const [dataset, setDataset] = useState<DatasetStatus | null>(null);
  const [history, setHistory] = useState<HospitalRunTelemetry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [connected, setConnected] = useState(false);
  const [lastEvent, setLastEvent] = useState<FederatedEventMessage | null>(null);

  const loadData = async () => {
    try {
      const [liveStatus, globalOverview, model, localDataset, trainingHistory] = await Promise.all([
        fetchHospitalLiveStatus(activeHospitalId),
        fetchFederatedOverview(),
        fetchHospitalCurrentModel(activeHospitalId),
        fetchHospitalDatasetStatus(activeHospitalId),
        fetchHospitalTrainingHistory(activeHospitalId),
      ]);
      setLive(liveStatus);
      setOverview(globalOverview);
      setCurrentModel(model);
      setDataset(localDataset);
      setHistory(trainingHistory);
      setError(null);
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "Unable to load federated learning status.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    setLoading(true);
    loadData();
  }, [activeHospitalId]);

  useEffect(() => {
    if (!hospitalCode) return;
    const unsubscribe = createFederatedWebSocket(
      (event) => {
        setLastEvent(event);
        if (!event.hospital_id || event.hospital_id === hospitalCode) {
          fetchHospitalLiveStatus(activeHospitalId).then(setLive).catch(() => undefined);
          if (["ROUND_COMPLETED", "ROUND_FAILED", "MODEL_DISTRIBUTED"].includes(event.event)) {
            loadData();
          }
        }
      },
      () => setConnected(false),
      hospitalCode,
      setConnected,
    );
    return unsubscribe;
  }, [activeHospitalId, hospitalCode]);

  useEffect(() => {
    if (!live || !activeStatuses.has(live.status) || ["READY", "COMPLETED", "FAILED"].includes(live.round_status ?? "")) return;
    const interval = window.setInterval(() => {
      fetchHospitalLiveStatus(activeHospitalId).then(setLive).catch(() => undefined);
    }, 2500);
    return () => window.clearInterval(interval);
  }, [activeHospitalId, live?.status]);

  const currentIndex = useMemo(() => statusIndex(live?.status ?? "WAITING"), [live?.status]);
  const isActive = Boolean(live && live.round_status && !["READY", "COMPLETED", "FAILED"].includes(live.round_status));
  const local = live?.local_training;
  const lastSync = live?.last_event_at ?? overview?.last_updated;

  return (
    <AppLayout
      role="hospital"
      title="Federated Learning"
      subtitle="Observe this hospital's participation in coordinator-managed training rounds."
    >
      <div className="space-y-6">
        {error && <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">{error}</div>}

        <section className="rounded-xl border border-[#DDE3DC] bg-white p-6 shadow-xs">
          <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
            <div>
              <div className="mb-2 flex items-center gap-2 text-xs text-[#101B16]/55">
                <span className={`h-2 w-2 rounded-full ${connected ? "bg-[#1F6F5C]" : "bg-[#B3261E]"}`} />
                {connected ? "Connected" : "WebSocket disconnected"}
              </div>
              <h2 className="font-serif text-2xl font-medium text-[#101B16]">{hospital?.name ?? "Hospital"}</h2>
              <p className="mt-1 font-mono text-xs text-[#101B16]/55">{hospitalCode ?? "Hospital identity loading"}</p>
            </div>
            <div className="text-left md:text-right">
              <p className="text-[11px] font-semibold uppercase tracking-wider text-[#101B16]/50">Round state</p>
              <p className={`mt-1 text-sm font-semibold ${live?.status === "FAILED" ? "text-[#B3261E]" : "text-[#1F6F5C]"}`}>
                {loading ? "Loading..." : isActive ? live?.status : "No active training round"}
              </p>
              {lastSync && <p className="mt-1 text-[11px] text-[#101B16]/45">Last event {new Date(lastSync).toLocaleString()}</p>}
            </div>
          </div>
          <div className="mt-5 grid grid-cols-2 gap-3 md:grid-cols-4">
            <Metric label="Current model" value={live?.global_model_version ?? currentModel?.current_model_version ?? "-"} mono />
            <Metric label="Round" value={live?.round ? `#${live.round}` : "-"} />
            <Metric label="Global accuracy" value={formatMetric(overview?.global_accuracy)} />
            <Metric label="Global F1" value={formatMetric(overview?.global_f1)} />
          </div>
        </section>

        <section className="rounded-xl border border-[#DDE3DC] bg-white p-6 shadow-xs">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-sm font-semibold text-[#101B16]">Your round lifecycle</h3>
              <p className="mt-1 text-xs text-[#101B16]/55">Stages reflect coordinator events; no training percentage is inferred.</p>
            </div>
            {live?.status === "FAILED" && <span className="rounded-full bg-red-50 px-3 py-1 text-xs font-semibold text-[#B3261E]">Unable to participate in this round</span>}
          </div>
          <div className="mt-5 grid gap-3 md:grid-cols-3 lg:grid-cols-6">
            {lifecycle.map((item, index) => {
              const complete = live?.status === "COMPLETED" || currentIndex > index;
              const current = currentIndex === index || (item.status === "MODEL_RECEIVED" && live?.status === "TRAINING_COMPLETED");
              return (
                <div key={item.status} className={`rounded-lg border p-3 ${complete ? "border-[#1F6F5C]/30 bg-[#EBF5F1]" : current ? "border-[#3B3F8C]/30 bg-[#F0F0FA]" : "border-[#DDE3DC] bg-[#F7F9F6]"}`}>
                  <div className={`mb-2 text-sm font-bold ${complete ? "text-[#1F6F5C]" : current ? "text-[#3B3F8C]" : "text-[#101B16]/35"}`}>{complete ? "✓" : current ? "●" : "○"}</div>
                  <p className="text-xs font-medium text-[#101B16]">{item.label}</p>
                </div>
              );
            })}
          </div>
          {lastEvent && <p className="mt-4 text-xs text-[#101B16]/55">Latest event: <span className="font-mono">{lastEvent.event}</span></p>}
        </section>

        <div className="grid gap-6 lg:grid-cols-2">
          <section className="rounded-xl border border-[#DDE3DC] bg-white p-6 shadow-xs">
            <h3 className="text-sm font-semibold text-[#101B16]">Your contribution <span className="ml-1 text-[10px] font-normal uppercase text-[#1F6F5C]">Local metrics</span></h3>
            <div className="mt-4 grid grid-cols-2 gap-3">
              <Metric label="Training samples" value={local?.samples?.toLocaleString() ?? "-"} />
              <Metric label="Training accuracy" value={formatMetric(local?.accuracy)} />
              <Metric label="Training F1" value={formatMetric(local?.f1)} />
              <Metric label="Training loss" value={formatMetric(local?.loss, false)} />
              <Metric label="Validation accuracy" value={formatMetric(history[0]?.val_acc)} />
              <Metric label="Training duration" value={local?.duration_sec == null ? "-" : `${local.duration_sec.toFixed(1)}s`} />
            </div>
          </section>

          <section className="rounded-xl border border-[#DDE3DC] bg-white p-6 shadow-xs">
            <h3 className="text-sm font-semibold text-[#101B16]">Global model <span className="ml-1 text-[10px] font-normal uppercase text-[#3B3F8C]">Global metrics</span></h3>
            <div className="mt-4 grid grid-cols-2 gap-3">
              <Metric label="Version" value={currentModel?.current_model_version ?? live?.global_model_version ?? "-"} mono />
              <Metric label="Global round" value={String(currentModel?.global_round ?? live?.round ?? "-")} />
              <Metric label="Accuracy" value={formatMetric(overview?.global_accuracy)} />
              <Metric label="F1" value={formatMetric(overview?.global_f1)} />
              <Metric label="Precision" value={formatMetric(overview?.global_precision)} />
              <Metric label="Recall" value={formatMetric(overview?.global_recall)} />
              <Metric label="Loss" value={formatMetric(overview?.global_loss, false)} />
              <Metric label="Dataset valid" value={dataset ? (dataset.is_valid ? "Validated" : "Invalid") : "-"} />
            </div>
          </section>
        </div>

        <section className="rounded-xl border border-[#DDE3DC] bg-white p-6 shadow-xs">
          <div className="flex flex-col gap-1 md:flex-row md:items-center md:justify-between">
            <div>
              <h3 className="text-sm font-semibold text-[#101B16]">Your training history</h3>
              <p className="text-xs text-[#101B16]/55">Only this hospital's persisted local telemetry is shown.</p>
            </div>
            <span className="text-xs text-[#101B16]/55">{history.length} round{history.length === 1 ? "" : "s"}</span>
          </div>
          {history.length === 0 ? (
            <div className="mt-4 rounded-lg border border-dashed border-[#DDE3DC] p-8 text-center text-sm text-[#101B16]/50">No completed hospital training rounds yet.</div>
          ) : (
            <div className="mt-4 overflow-x-auto"><table className="w-full text-left text-xs"><thead className="border-b border-[#DDE3DC] text-[10px] uppercase tracking-wider text-[#101B16]/50"><tr><th className="px-3 py-2">Round</th><th className="px-3 py-2">Model</th><th className="px-3 py-2">Samples</th><th className="px-3 py-2">Local accuracy</th><th className="px-3 py-2">Local F1</th><th className="px-3 py-2">Status</th><th className="px-3 py-2">Date</th></tr></thead><tbody>{history.map((run) => <tr key={run.id} className="border-b border-[#DDE3DC]/60"><td className="px-3 py-3 font-semibold">#{run.round_number}</td><td className="px-3 py-3 font-mono text-[10px]">{run.model_version ?? "-"}</td><td className="px-3 py-3">{run.sample_count.toLocaleString()}</td><td className="px-3 py-3">{formatMetric(run.train_acc)}</td><td className="px-3 py-3">{formatMetric(run.train_f1)}</td><td className="px-3 py-3 text-[#1F6F5C]">Completed</td><td className="px-3 py-3 text-[#101B16]/55">{new Date(run.created_at).toLocaleDateString()}</td></tr>)}</tbody></table></div>
          )}
        </section>

        <div className="grid gap-6 lg:grid-cols-2">
          <section className="rounded-xl border border-[#DDE3DC] bg-white p-6 shadow-xs">
            <h3 className="text-sm font-semibold text-[#101B16]">Federated network</h3>
            <div className="mt-4 flex items-center justify-between rounded-lg bg-[#F7F9F6] p-4"><div><p className="text-2xl font-semibold text-[#101B16]">{overview?.active_hospitals_count ?? "-"}</p><p className="text-xs text-[#101B16]/55">hospitals participating</p></div><div className="text-right"><p className="text-xs font-semibold text-[#1F6F5C]">Your hospital</p><p className="text-xs text-[#101B16]/55">{live?.status === "COMPLETED" ? "✓ Completed" : live?.status ?? "Waiting"}</p></div></div>
          </section>
          <section className="rounded-xl border border-[#DDE3DC] bg-white p-6 shadow-xs">
            <h3 className="text-sm font-semibold text-[#101B16]">How federated learning works</h3>
            <div className="mt-4 flex flex-wrap items-center gap-2 text-xs font-medium text-[#101B16]"><span className="rounded-md bg-[#EBF5F1] px-3 py-2">Your patient data</span><span>→</span><span className="rounded-md bg-[#EBF5F1] px-3 py-2">Local training</span><span>→</span><span className="rounded-md bg-[#F0F0FA] px-3 py-2">Model update</span><span>→</span><span className="rounded-md bg-[#F0F0FA] px-3 py-2">FedAvg</span><span>→</span><span className="rounded-md bg-[#EBF5F1] px-3 py-2">Global model</span></div>
            <p className="mt-4 text-xs leading-5 text-[#101B16]/60">Patient data remains within your hospital. Only model updates and training telemetry are contributed to the coordinator-backed aggregation workflow.</p>
          </section>
        </div>
      </div>
    </AppLayout>
  );
}

function Metric({ label, value, mono = false }: { label: string; value: string; mono?: boolean }) {
  return <div className="rounded-lg border border-[#DDE3DC] bg-[#F7F9F6] p-3"><p className="text-[10px] font-semibold uppercase tracking-wider text-[#101B16]/50">{label}</p><p className={`mt-1 truncate text-sm font-semibold text-[#101B16] ${mono ? "font-mono text-xs" : ""}`}>{value}</p></div>;
}
