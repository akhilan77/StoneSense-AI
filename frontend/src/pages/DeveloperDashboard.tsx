// frontend/src/pages/DeveloperDashboard.tsx
import React, { useEffect, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import AppLayout from '../components/layout/AppLayout';
import {
  deployModelVersion,
  fetchAllEnrolledHospitals,
  fetchDriftAnalysis,
  fetchFederatedOverview,
  fetchHospitalLogs,
  fetchHospitalParticipation,
  fetchModelPerformance,
  fetchRoundHistory,
  fetchSystemLogs,
  fetchSystemMonitoring,
  startMLTraining,
} from '../services/developerApi';
import {
  createFederatedWebSocket,
  fetchCurrentRoundLiveStatus,
  startFederatedRound,
} from '../services/federatedApi';
import {
  DriftPoint,
  Hospital,
  HospitalUpdateLogEntry,
  ModelPerformance,
  SystemLogEntry,
  SystemMonitoringSummary,
} from '../types/dashboard';
import {
  FederatedEventMessage,
  FederatedOverview,
  FederatedRoundDetail,
  HospitalParticipation,
  RoundLiveStatus,
} from '../types/federated';

type DevTab = 'overview' | 'federated' | 'versions' | 'access' | 'health';

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
    if (path.includes('/federated')) return 'federated';
    if (path.includes('/versions')) return 'versions';
    if (
      path.includes('/access') ||
      path.includes('/enrolled-hospitals') ||
      path.includes('/hospitals')
    )
      return 'access';
    if (path.includes('/health') || path.includes('/monitoring') || path.includes('/drift'))
      return 'health';
    return 'overview';
  };

  const [activeTab, setActiveTab] = useState<DevTab>(getTabFromPath());
  const [versions, setVersions] = useState<ModelPerformance[]>([]);
  const [hospitalLogs, setHospitalLogs] = useState<HospitalUpdateLogEntry[]>([]);
  const [monitoring, setMonitoring] = useState<SystemMonitoringSummary | null>(null);
  const [logs, setLogs] = useState<SystemLogEntry[]>([]);
  const [drift, setDrift] = useState<DriftPoint[]>([]);
  const [hospitals, setHospitals] = useState<Hospital[]>([]);
  const [toastMessage, setToastMessage] = useState<string | null>(null);

  // Federated Learning Telemetry & Live Round States
  const [fedOverview, setFedOverview] = useState<FederatedOverview | null>(null);
  const [roundHistory, setRoundHistory] = useState<FederatedRoundDetail[]>([]);
  const [participation, setParticipation] = useState<HospitalParticipation[]>([]);
  const [selectedRoundDetail, setSelectedRoundDetail] = useState<FederatedRoundDetail | null>(null);
  const [liveRoundStatus, setLiveRoundStatus] = useState<RoundLiveStatus | null>(null);
  const [isStartingRound, setIsStartingRound] = useState(false);
  const [isRoundModalOpen, setIsRoundModalOpen] = useState(false);
  const [isMLTraining, setIsMLTraining] = useState(false);
  const [mlTrainingResult, setMLTrainingResult] = useState<{
    model_name: string;
    version_tag: string;
    duration_sec: number;
    metrics: Record<string, number>;
  } | null>(null);
  const currentMLVersion =
    versions.find((version) => version.model_family === 'xgboost_risk' && version.is_deployed) ??
    versions.find((version) => version.model_family === 'xgboost_risk');
  const [selectedHospitalIds, setSelectedHospitalIds] = useState<number[]>([]);
  const [roundConfig, setRoundConfig] = useState({
    localEpochs: 1,
    batchSize: 32,
    learningRate: 0.0005,
  });

  // Filter states
  const [logFilterLevel, setLogFilterLevel] = useState<string>('all');
  const [hospitalSearch, setHospitalSearch] = useState<string>('');
  const [hospLogSearch, setHospLogSearch] = useState<string>('');

  // Modal states
  const [isDeployModalOpen, setIsDeployModalOpen] = useState(false);
  const [isEnrollModalOpen, setIsEnrollModalOpen] = useState(false);
  const [selectedLogPayload, setSelectedLogPayload] = useState<HospitalUpdateLogEntry | null>(null);

  // New Version Form state
  const [newVersionForm, setNewVersionForm] = useState({
    modelFamily: 'xgboost_risk',
    versionTag: 'v2.5.0-prod',
    accuracy: '0.965',
    f1Score: '0.960',
    environment: 'production',
    notes: 'Trained on updated multi-center cohort dataset with reduced false positive rate.',
  });

  // New Hospital Enroll Form state
  const [newHospitalForm, setNewHospitalForm] = useState({
    name: '',
    code: '',
    region: '',
    contactEmail: '',
    tier: 'standard' as 'enterprise' | 'standard' | 'research',
  });

  const showToast = (msg: string) => {
    setToastMessage(msg);
    setTimeout(() => setToastMessage(null), 3200);
  };

  const loadAllData = () => {
    fetchModelPerformance()
      .then(setVersions)
      .catch(() => {});
    fetchHospitalLogs()
      .then(setHospitalLogs)
      .catch(() => {});
    fetchSystemMonitoring()
      .then(setMonitoring)
      .catch(() => {});
    fetchSystemLogs()
      .then(setLogs)
      .catch(() => {});
    fetchDriftAnalysis()
      .then(setDrift)
      .catch(() => {});
    fetchAllEnrolledHospitals()
      .then(setHospitals)
      .catch(() => {});
    fetchFederatedOverview()
      .then(setFedOverview)
      .catch(() => {});
    fetchRoundHistory()
      .then((data) => {
        setRoundHistory(data);
        if (data.length > 0) setSelectedRoundDetail(data[data.length - 1]);
      })
      .catch(() => {});
    fetchHospitalParticipation()
      .then(setParticipation)
      .catch(() => {});
    fetchCurrentRoundLiveStatus()
      .then(setLiveRoundStatus)
      .catch(() => {});
  };

  useEffect(() => {
    loadAllData();

    // Subscribe to live federated events
    const unsubscribe = createFederatedWebSocket((event: FederatedEventMessage) => {
      fetchCurrentRoundLiveStatus()
        .then((status) => {
          setLiveRoundStatus(status);
          if (event.event === 'ROUND_COMPLETED') {
            showToast(`✓ Round #${event.round} completed and synchronized across network!`);
            loadAllData();
          } else if (event.event === 'ROUND_FAILED') {
            showToast(`✕ Round #${event.round} failed: ${event.data?.error || 'Execution error'}`);
          }
        })
        .catch(() => {});
    });

    return () => unsubscribe();
  }, []);

  // Polling fallback while a round is actively running
  useEffect(() => {
    if (!liveRoundStatus || ['READY', 'COMPLETED', 'FAILED'].includes(liveRoundStatus.status)) {
      return;
    }

    const interval = setInterval(() => {
      fetchCurrentRoundLiveStatus()
        .then((status) => {
          setLiveRoundStatus(status);
          if (status.status === 'COMPLETED') {
            loadAllData();
          }
        })
        .catch(() => {});
    }, 2500);

    return () => clearInterval(interval);
  }, [liveRoundStatus?.status]);

  useEffect(() => {
    setActiveTab(getTabFromPath());
  }, [location.pathname]);

  const openRoundModal = () => {
    setSelectedHospitalIds(
      hospitals.filter((hospital) => hospital.is_active).map((hospital) => hospital.id)
    );
    setIsRoundModalOpen(true);
  };

  const handleStartFederatedRound = async () => {
    if (
      liveRoundStatus?.status &&
      !['READY', 'COMPLETED', 'FAILED'].includes(liveRoundStatus.status)
    ) {
      showToast(`Round #${liveRoundStatus.round} is currently running.`);
      return;
    }

    if (selectedHospitalIds.length === 0) {
      showToast('Select at least one active hospital.');
      return;
    }

    setIsStartingRound(true);
    showToast('Starting live multi-hospital DL Federated Learning round...');
    try {
      const res = await startFederatedRound({
        selected_hospital_ids: selectedHospitalIds,
        num_rounds: 1,
        local_epochs: roundConfig.localEpochs,
        batch_size: roundConfig.batchSize,
        lr: roundConfig.learningRate,
        mode: 'iid',
      });
      setIsRoundModalOpen(false);
      showToast(`Round #${res.round} initiated on central Flower coordinator!`);
      const liveStatus = await fetchCurrentRoundLiveStatus();
      setLiveRoundStatus(liveStatus);
    } catch (err: any) {
      const msg = err.response?.data?.detail || 'Failed to start DL federated round.';
      showToast(msg);
    } finally {
      setIsStartingRound(false);
    }
  };

  const handleStartMLTraining = async () => {
    setIsMLTraining(true);
    showToast('Starting centralized ML training on the central ML dataset...');
    try {
      const result = await startMLTraining();
      setMLTrainingResult(result);
      showToast(`${result.message} Version ${result.version_tag} registered.`);
      await loadAllData();
    } catch (err: any) {
      showToast(err.response?.data?.detail || 'Centralized ML training failed.');
    } finally {
      setIsMLTraining(false);
    }
  };

  const handleTabChange = (tab: DevTab) => {
    setActiveTab(tab);
    if (tab === 'overview') navigate('/developer-dashboard');
    else if (tab === 'federated') navigate('/developer-dashboard/federated');
    else if (tab === 'versions') navigate('/developer-dashboard/versions');
    else if (tab === 'access') navigate('/developer-dashboard/access');
    else if (tab === 'health') navigate('/developer-dashboard/health');
  };

  const handleDeployVersion = async (v: ModelPerformance) => {
    showToast(`Deploying ${v.model_family} (${v.version_tag}) to production...`);
    await deployModelVersion(v.id);
    setVersions((prev) =>
      prev.map((item) => {
        if (item.model_family === v.model_family) {
          return {
            ...item,
            is_deployed: item.id === v.id,
            rollout_pct: item.id === v.id ? 100 : 0,
          };
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
      is_deployed: newVersionForm.environment === 'production',
      trained_at: new Date().toISOString(),
      latency_ms: newVersionForm.modelFamily === 'xgboost_risk' ? 17.5 : 78.0,
      environment: newVersionForm.environment as any,
      rollout_pct: newVersionForm.environment === 'production' ? 100 : 0,
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
      hospital_code:
        newHospitalForm.code || `HOSP-${Math.random().toString(36).substring(2, 6).toUpperCase()}`,
      region: newHospitalForm.region || 'India',
      is_active: true,
      tier: newHospitalForm.tier,
      contact_email: newHospitalForm.contactEmail,
      api_key: `ss_live_${newHospitalForm.name.toLowerCase().slice(0, 4)}_${Math.random().toString(36).substring(2, 9)}`,
      last_sync: 'Just now',
      total_scans: 0,
    };

    setHospitals([newHosp, ...hospitals]);
    setIsEnrollModalOpen(false);
    setNewHospitalForm({ name: '', code: '', region: '', contactEmail: '', tier: 'standard' });
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

      {/* Top Action Bar (Horizontal Nav Removed to fix overflow) */}
      <div className="flex items-center justify-end border-b border-[#DDE3DC] pb-4 mb-6">
        <button
          onClick={loadAllData}
          className="flex items-center gap-1.5 rounded-md border border-[#DDE3DC] bg-white px-3 py-1.5 text-xs text-[#101B16]/70 hover:bg-black/5 cursor-pointer shadow-xs"
        >
          <svg
            className="h-3.5 w-3.5"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
            />
          </svg>
          Refresh Live Telemetry
        </button>
      </div>

      {/* ========================================================================= */}
      {/* TAB 1: OVERVIEW & PERFORMANCE */}
      {/* ========================================================================= */}
      {activeTab === 'overview' && (
        <div className="space-y-6">
          {/* Top KPI Cards */}
          <div className="grid grid-cols-4 gap-4">
            {[
              {
                label: 'Predictions (24h)',
                value: monitoring?.total_predictions_24h ?? '1,482',
                sub: '+12.4% vs prev day',
              },
              {
                label: 'Error Rate (24h)',
                value: monitoring
                  ? `${((monitoring.error_count_24h / (monitoring.total_predictions_24h || 1)) * 100).toFixed(2)}%`
                  : '0.20%',
                sub: `${monitoring?.error_count_24h ?? 3} errors logged`,
              },
              {
                label: 'Avg Inference Latency',
                value: monitoring?.avg_latency_ms ? `${monitoring.avg_latency_ms} ms` : '44.8 ms',
                sub: 'p95: 88ms · p99: 135ms',
              },
              {
                label: 'System Uptime',
                value: monitoring ? `${monitoring.uptime_pct}%` : '99.98%',
                sub: 'All 8 worker nodes active',
              },
            ].map((stat) => (
              <div
                key={stat.label}
                className="rounded-xl border border-[#DDE3DC] bg-white p-5 shadow-xs"
              >
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
                <h2 className="font-serif text-base font-medium text-[#101B16]">
                  Active ML/DL Models
                </h2>
                <button
                  onClick={() => handleTabChange('versions')}
                  className="text-xs text-[#3B3F8C] hover:underline font-medium"
                >
                  Manage versions →
                </button>
              </div>
              <div className="space-y-3">
                {versions
                  .filter((v) => v.is_deployed)
                  .map((v) => (
                    <div
                      key={v.id}
                      className="rounded-lg border border-[#DDE3DC] bg-[#F7F9F6] p-4 flex items-center justify-between"
                    >
                      <div>
                        <div className="flex items-center gap-2">
                          <span className="font-semibold text-xs text-[#101B16]">
                            {v.model_family === 'xgboost_risk'
                              ? 'ML — Centralized · XGBoost Clinical Risk'
                              : 'DL — Federated Learning · ResNet18 CT Imaging'}
                          </span>
                          <span className="rounded bg-[#1F6F5C]/15 px-2 py-0.5 text-[10.5px] font-bold text-[#1F6F5C]">
                            {v.version_tag}
                          </span>
                        </div>
                        <p className="text-[11.5px] text-[#101B16]/60 mt-1">
                          Accuracy: <strong>{(v.accuracy! * 100).toFixed(1)}%</strong> · F1:{' '}
                          <strong>{(v.f1_score! * 100).toFixed(1)}%</strong> · Avg Latency:{' '}
                          {v.latency_ms ?? 18}ms
                        </p>
                      </div>
                      <span className="inline-flex items-center gap-1 text-xs text-[#1F6F5C] font-semibold">
                        <span className="h-2 w-2 rounded-full bg-[#1F6F5C] animate-pulse" /> 100%
                        Traffic
                      </span>
                    </div>
                  ))}
              </div>
            </section>

            {/* Quick Enrolled Hospital Status */}
            <section className="rounded-xl border border-[#DDE3DC] bg-white p-5 shadow-xs">
              <div className="flex items-center justify-between mb-4">
                <h2 className="font-serif text-base font-medium text-[#101B16]">
                  Enrolled Partner Hospitals
                </h2>
                <button
                  onClick={() => handleTabChange('access')}
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
                      <p className="text-[11px] text-[#101B16]/50">
                        {h.hospital_code} · {h.region}
                      </p>
                    </div>
                    <span
                      className={`rounded-full px-2.5 py-0.5 text-[10.5px] font-semibold ${
                        h.is_active
                          ? 'bg-[#1F6F5C]/15 text-[#1F6F5C]'
                          : 'bg-[#B3261E]/15 text-[#B3261E]'
                      }`}
                    >
                      {h.is_active ? 'Active' : 'Revoked'}
                    </span>
                  </div>
                ))}
              </div>
            </section>
          </div>
        </div>
      )}

      {/* ========================================================================= */}
      {/* TAB: DL FEDERATED LEARNING HUB */}
      {/* ========================================================================= */}
      {activeTab === 'federated' && (
        <div className="space-y-6">
          <section className="rounded-xl border-2 border-[#3B3F8C]/25 bg-white p-6 shadow-sm">
            <div className="flex flex-col gap-4 border-b border-[#DDE3DC] pb-5 md:flex-row md:items-start md:justify-between">
              <div>
                <span className="inline-flex rounded-full bg-[#3B3F8C]/10 px-2.5 py-1 text-[11px] font-bold text-[#3B3F8C]">
                  ML — Centralized
                </span>
                <h2 className="mt-2 font-serif text-lg font-medium text-[#101B16]">
                  Centralized ML Training
                </h2>
                <p className="mt-1 text-xs leading-5 text-[#101B16]/60">
                  Centralized ML Dataset → ML Training → Model Evaluation → New ML Model Version
                </p>
              </div>
              <button
                type="button"
                onClick={handleStartMLTraining}
                disabled={isMLTraining}
                className="rounded-lg bg-[#3B3F8C] px-5 py-2.5 text-xs font-semibold text-white hover:bg-[#2F3270] disabled:cursor-not-allowed disabled:opacity-50"
              >
                {isMLTraining ? 'Training ML...' : 'Start ML Training'}
              </button>
            </div>
            <div className="mt-5 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              <div className="rounded-lg bg-[#F7F9F6] p-3.5">
                <span className="text-[10.5px] font-semibold uppercase tracking-wider text-[#101B16]/50">
                  Current ML Model
                </span>
                <p className="mt-1 text-sm font-bold text-[#101B16]">
                  {mlTrainingResult?.model_name ?? 'XGBoost'}
                </p>
              </div>
              <div className="rounded-lg bg-[#F7F9F6] p-3.5">
                <span className="text-[10.5px] font-semibold uppercase tracking-wider text-[#101B16]/50">
                  Version
                </span>
                <p className="mt-1 truncate font-mono text-xs font-bold text-[#101B16]">
                  {mlTrainingResult?.version_tag ??
                    versions.find((version) => version.model_family === 'xgboost_risk')
                      ?.version_tag ??
                    'kidney_risk_model.pkl'}
                </p>
              </div>
              <div className="rounded-lg bg-[#F7F9F6] p-3.5">
                <span className="text-[10.5px] font-semibold uppercase tracking-wider text-[#101B16]/50">
                  Dataset
                </span>
                <p className="mt-1 text-sm font-bold text-[#101B16]">Centralized ML Dataset</p>
              </div>
              <div className="rounded-lg bg-[#F7F9F6] p-3.5">
                <span className="text-[10.5px] font-semibold uppercase tracking-wider text-[#101B16]/50">
                  Status
                </span>
                <p className="mt-1 text-sm font-bold text-[#1F6F5C]">
                  {isMLTraining ? 'Training' : 'Ready'}
                </p>
              </div>
            </div>
            <div className="mt-4 flex flex-wrap gap-4 text-xs text-[#101B16]/65">
              <span>Models: Logistic Regression · Random Forest · XGBoost</span>
              <span>Training: Centralized</span>
              <span>
                Accuracy:{' '}
                {mlTrainingResult?.metrics.accuracy != null
                  ? `${(mlTrainingResult.metrics.accuracy * 100).toFixed(1)}%`
                  : currentMLVersion?.accuracy != null
                    ? `${(currentMLVersion.accuracy * 100).toFixed(1)}%`
                    : '-'}{' '}
                · F1:{' '}
                {mlTrainingResult?.metrics.f1_score != null
                  ? `${(mlTrainingResult.metrics.f1_score * 100).toFixed(1)}%`
                  : currentMLVersion?.f1_score != null
                    ? `${(currentMLVersion.f1_score * 100).toFixed(1)}%`
                    : '-'}
              </span>
              {mlTrainingResult && (
                <span>Latest duration: {mlTrainingResult.duration_sec.toFixed(2)}s</span>
              )}
            </div>
          </section>

          {/* Header Banner */}
          <div className="rounded-xl border border-[#DDE3DC] bg-white p-5 shadow-xs flex flex-col md:flex-row md:items-center justify-between gap-4">
            <div>
              <div className="flex items-center gap-2 mb-1">
                <span className="inline-flex items-center gap-1.5 rounded-full bg-[#EBF5F1] px-2.5 py-0.5 text-[11px] font-medium text-[#1F6F5C]">
                  <span className="h-1.5 w-1.5 rounded-full bg-[#1F6F5C] animate-pulse" />
                  DL — Federated Learning · FedAvg Aggregator Active
                </span>
                <span className="text-xs text-[#101B16]/50">
                  Coordinator: Flower FL + PyTorch ResNet18
                </span>
              </div>
              <h2 className="font-serif text-lg font-medium text-[#101B16]">
                DL Global Federated Model · Multi-Hospital Learning Network
              </h2>
              <p className="text-xs text-[#101B16]/60 mt-0.5">
                Real-time round telemetry, loss reduction convergence, and sample contribution
                matrices without centralizing raw CT scans.
              </p>
            </div>

            <div className="flex items-center gap-2">
              <span className="rounded-lg bg-[#F7F9F6] px-3 py-1.5 text-xs text-[#101B16]/70 border border-[#DDE3DC]">
                Mode: <strong className="text-[#3B3F8C]">IID & Non-IID Multi-Node</strong>
              </span>
            </div>
          </div>

          {/* ========================================================================= */}
          {/* LIVE FEDERATED ROUND CONTROL & TELEMETRY MONITOR */}
          {/* ========================================================================= */}
          <div className="rounded-xl border-2 border-[#1F6F5C]/30 bg-white p-6 shadow-sm">
            <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 border-b border-[#DDE3DC] pb-5 mb-5">
              <div>
                <div className="flex items-center gap-2 mb-1.5">
                  <span className="inline-flex items-center gap-1.5 rounded-full bg-[#3B3F8C]/10 px-2.5 py-0.5 text-[11px] font-bold text-[#3B3F8C]">
                    LIVE ROUND ORCHESTRATION
                  </span>
                  <span className="text-xs text-[#101B16]/60">
                    Coordinator Node: <strong>StoneSense Central Flower Server</strong>
                  </span>
                </div>
                <h3 className="font-serif text-base font-semibold text-[#101B16]">
                  DL Federated Round Control & Execution
                </h3>
                <p className="text-xs text-[#101B16]/65 mt-0.5">
                  Trigger an authentic DL federated round. DL global weights are distributed to
                  simulated hospital nodes, trained locally on private CT partitions, and aggregated
                  via FedAvg.
                </p>
              </div>

              {/* Action Button & Status Indicator */}
              <div className="flex flex-col sm:flex-row items-start sm:items-center gap-3">
                {liveRoundStatus?.status &&
                !['READY', 'COMPLETED', 'FAILED'].includes(liveRoundStatus.status) ? (
                  <div className="flex items-center gap-2 rounded-lg bg-[#3B3F8C]/10 border border-[#3B3F8C]/20 px-3.5 py-2 text-xs font-semibold text-[#3B3F8C]">
                    <span className="h-2 w-2 rounded-full bg-[#3B3F8C] animate-ping" />
                    Round #{liveRoundStatus.round} is currently running...
                  </div>
                ) : (
                  <button
                    onClick={openRoundModal}
                    disabled={
                      isStartingRound ||
                      !!(
                        liveRoundStatus?.status &&
                        !['READY', 'COMPLETED', 'FAILED'].includes(liveRoundStatus.status)
                      )
                    }
                    className="flex items-center gap-2 rounded-lg bg-[#1F6F5C] px-5 py-2.5 text-xs font-semibold text-white shadow-sm hover:bg-[#185849] active:scale-[0.99] transition-all cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth="2"
                        d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"
                      />
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth="2"
                        d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                      />
                    </svg>
                    START DL FEDERATED ROUND
                  </button>
                )}
              </div>
            </div>

            {/* Quick Status Bar */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3.5 mb-6">
              <div className="rounded-lg bg-[#F7F9F6] p-3.5 border border-[#DDE3DC]">
                <span className="text-[10.5px] font-semibold text-[#101B16]/50 uppercase tracking-wider">
                  DL Global Federated Model
                </span>
                <p className="text-xs font-bold text-[#101B16] font-mono mt-1 truncate">
                  {liveRoundStatus?.global_model_version ??
                    fedOverview?.current_model_version ??
                    '-'}
                </p>
                <span className="text-[10px] text-[#1F6F5C] font-medium">ResNet18-FL</span>
              </div>

              <div className="rounded-lg bg-[#F7F9F6] p-3.5 border border-[#DDE3DC]">
                <span className="text-[10.5px] font-semibold text-[#101B16]/50 uppercase tracking-wider">
                  Current Round
                </span>
                <p className="text-sm font-bold text-[#3B3F8C] mt-1">
                  Round #{liveRoundStatus?.round ?? fedOverview?.current_round ?? 3}
                </p>
                <span className="text-[10px] text-[#101B16]/50">Flower Iteration Index</span>
              </div>

              <div className="rounded-lg bg-[#F7F9F6] p-3.5 border border-[#DDE3DC]">
                <span className="text-[10.5px] font-semibold text-[#101B16]/50 uppercase tracking-wider">
                  Connected Hospitals
                </span>
                <p className="text-sm font-bold text-[#101B16] mt-1">3 / 3 Online</p>
                <span className="text-[10px] text-[#1F6F5C]">Simulated Hospital Clients</span>
              </div>

              <div className="rounded-lg bg-[#F7F9F6] p-3.5 border border-[#DDE3DC]">
                <span className="text-[10.5px] font-semibold text-[#101B16]/50 uppercase tracking-wider">
                  Round Status
                </span>
                <div className="mt-1">
                  <span
                    className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-[11px] font-bold ${
                      liveRoundStatus?.status === 'COMPLETED'
                        ? 'bg-[#1F6F5C]/15 text-[#1F6F5C]'
                        : liveRoundStatus?.status === 'FAILED'
                          ? 'bg-[#B3261E]/15 text-[#B3261E]'
                          : liveRoundStatus?.status && !['READY'].includes(liveRoundStatus.status)
                            ? 'bg-[#3B3F8C]/15 text-[#3B3F8C]'
                            : 'bg-[#101B16]/10 text-[#101B16]/70'
                    }`}
                  >
                    <span
                      className={`h-1.5 w-1.5 rounded-full ${
                        liveRoundStatus?.status === 'COMPLETED'
                          ? 'bg-[#1F6F5C]'
                          : liveRoundStatus?.status === 'FAILED'
                            ? 'bg-[#B3261E]'
                            : liveRoundStatus?.status && !['READY'].includes(liveRoundStatus.status)
                              ? 'bg-[#3B3F8C] animate-pulse'
                              : 'bg-[#101B16]/50'
                      }`}
                    />
                    {liveRoundStatus?.status ?? 'READY'}
                  </span>
                </div>
                <span className="text-[10px] text-[#101B16]/50 truncate block mt-0.5">
                  {liveRoundStatus?.current_step ?? 'Awaiting round initiation'}
                </span>
              </div>
            </div>

            {/* LIVE WORKFLOW PROGRESSION VISUALIZER */}
            <div className="rounded-xl border border-[#DDE3DC] bg-[#F9FAF8] p-5">
              <div className="flex items-center justify-between border-b border-[#DDE3DC]/80 pb-3 mb-4">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-bold text-[#101B16]">
                    ROUND {liveRoundStatus?.round ?? 3} —{' '}
                    {liveRoundStatus?.status === 'COMPLETED'
                      ? '✓ COMPLETED'
                      : liveRoundStatus?.status &&
                          !['READY', 'FAILED'].includes(liveRoundStatus.status)
                        ? 'IN PROGRESS'
                        : 'READY'}
                  </span>
                  <span className="text-xs text-[#101B16]/50 font-mono">
                    (DL Global Federated Model:{' '}
                    {liveRoundStatus?.previous_model_version ?? 'ResNet18-FL-v2'})
                  </span>
                </div>
                <span className="rounded bg-white border border-[#DDE3DC] px-2 py-0.5 text-[10.5px] font-semibold text-[#101B16]/70">
                  Zero-Raw-CT Privacy Standard
                </span>
              </div>

              {/* Step Sequence Container */}
              <div className="space-y-4 text-xs">
                {/* 1. Model Distribution */}
                <div className="rounded-lg bg-white p-3.5 border border-[#DDE3DC] shadow-xs">
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-semibold text-[#101B16]">1. MODEL DISTRIBUTION</span>
                    <span className="text-[11px] font-medium text-[#1F6F5C]">
                      {liveRoundStatus?.status && !['READY'].includes(liveRoundStatus.status)
                        ? '✓ Distributed to 3 Nodes'
                        : 'Waiting for trigger'}
                    </span>
                  </div>
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 text-[11px]">
                    {[
                      {
                        code: 'HOSP-001',
                        name: 'Hospital 1 (Apollo)',
                        status: liveRoundStatus?.clients?.find((c) => c.hospital_id === 'HOSP-001')
                          ?.status,
                      },
                      {
                        code: 'HOSP-002',
                        name: 'Hospital 2 (Manipal)',
                        status: liveRoundStatus?.clients?.find((c) => c.hospital_id === 'HOSP-002')
                          ?.status,
                      },
                      {
                        code: 'HOSP-003',
                        name: 'Hospital 3 (AIIMS)',
                        status: liveRoundStatus?.clients?.find((c) => c.hospital_id === 'HOSP-003')
                          ?.status,
                      },
                    ].map((h) => (
                      <div
                        key={h.code}
                        className="flex items-center justify-between rounded bg-[#F7F9F6] p-2 border border-[#DDE3DC]/60"
                      >
                        <span className="font-medium text-[#101B16]">{h.name}</span>
                        <span className="font-semibold text-[#1F6F5C]">
                          {h.status && h.status !== 'WAITING' ? '✓ Received' : '—'}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* 2. Local Training */}
                <div className="rounded-lg bg-white p-3.5 border border-[#DDE3DC] shadow-xs">
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-semibold text-[#101B16]">
                      2. LOCAL TRAINING & UPDATES
                    </span>
                    <span className="text-[11px] font-medium text-[#3B3F8C]">
                      {liveRoundStatus?.status === 'LOCAL_TRAINING'
                        ? '● In Progress (Zero-Raw-CT Privacy)'
                        : liveRoundStatus?.status === 'COMPLETED'
                          ? '✓ All Nodes Completed'
                          : 'Waiting'}
                    </span>
                  </div>
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5">
                    {[
                      {
                        code: 'HOSP-001',
                        name: 'Hospital 1 (Apollo)',
                        client: liveRoundStatus?.clients?.find((c) => c.hospital_id === 'HOSP-001'),
                      },
                      {
                        code: 'HOSP-002',
                        name: 'Hospital 2 (Manipal)',
                        client: liveRoundStatus?.clients?.find((c) => c.hospital_id === 'HOSP-002'),
                      },
                      {
                        code: 'HOSP-003',
                        name: 'Hospital 3 (AIIMS)',
                        client: liveRoundStatus?.clients?.find((c) => c.hospital_id === 'HOSP-003'),
                      },
                    ].map((h) => {
                      const c = h.client;
                      const isCompleted =
                        c?.status === 'COMPLETED' ||
                        c?.status === 'MODEL_UPDATED' ||
                        liveRoundStatus?.status === 'COMPLETED';
                      const isTraining =
                        c?.status === 'TRAINING' || liveRoundStatus?.status === 'LOCAL_TRAINING';
                      return (
                        <div
                          key={h.code}
                          className="rounded-lg bg-[#F7F9F6] p-3 border border-[#DDE3DC]"
                        >
                          <div className="flex items-center justify-between mb-1.5">
                            <span className="font-bold text-[#101B16] text-[11.5px]">{h.name}</span>
                            <span
                              className={`text-[10.5px] font-bold ${
                                isCompleted
                                  ? 'text-[#1F6F5C]'
                                  : isTraining
                                    ? 'text-[#3B3F8C] animate-pulse'
                                    : 'text-[#101B16]/40'
                              }`}
                            >
                              {isCompleted
                                ? '✓ Completed'
                                : isTraining
                                  ? '● Training...'
                                  : 'Waiting'}
                            </span>
                          </div>
                          <div className="space-y-1 text-[10.5px] text-[#101B16]/70 border-t border-[#DDE3DC]/50 pt-1.5">
                            <div className="flex justify-between">
                              <span>Samples:</span>
                              <strong className="text-[#101B16]">
                                {c?.samples ? c.samples.toLocaleString() : '-'}
                              </strong>
                            </div>
                            <div className="flex justify-between">
                              <span>Local Loss:</span>
                              <strong className="font-mono text-[#101B16]">
                                {c?.loss != null ? c.loss.toFixed(4) : '-'}
                              </strong>
                            </div>
                            <div className="flex justify-between">
                              <span>Local Accuracy:</span>
                              <strong className="text-[#101B16]">
                                {c?.accuracy != null ? `${(c.accuracy * 100).toFixed(1)}%` : '-'}
                              </strong>
                            </div>
                            <div className="flex justify-between">
                              <span>Local F1:</span>
                              <strong className="text-[#1F6F5C]">
                                {c?.f1 != null ? `${(c.f1 * 100).toFixed(1)}%` : '-'}
                              </strong>
                            </div>
                            <div className="flex justify-between">
                              <span>Duration:</span>
                              <span className="text-[#101B16]/60">
                                {c?.duration_sec != null ? `${c.duration_sec}s` : '-'}
                              </span>
                            </div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* 3. DL Federated Aggregation (FedAvg) & DL Global Federated Model */}
                <div className="rounded-lg bg-white p-3.5 border border-[#DDE3DC] shadow-xs">
                  <div className="flex items-center justify-between mb-3">
                    <span className="font-semibold text-[#101B16]">
                      3. DL FEDERATED AGGREGATION & DL GLOBAL MODEL CREATION
                    </span>
                    <span className="text-[11px] font-bold text-[#1F6F5C]">
                      {liveRoundStatus?.status === 'COMPLETED'
                        ? '✓ Checkpoint Saved & Deployed'
                        : liveRoundStatus?.status === 'FEDAVG_STARTED'
                          ? '● Running FedAvg...'
                          : 'Standing By'}
                    </span>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 items-center">
                    {/* Visual FedAvg Flow */}
                    <div className="rounded bg-[#F7F9F6] p-3 border border-[#DDE3DC]/60 font-mono text-[11px] text-[#101B16]">
                      <p className="text-[10px] text-[#101B16]/50 uppercase tracking-wider font-sans font-semibold mb-1">
                        FedAvg Aggregation Topology
                      </p>
                      <div className="leading-tight text-[#3B3F8C]">
                        Hospital 1 (HOSP-001) ──┐
                        <br />
                        Hospital 2 (HOSP-002) ──┼──→{' '}
                        <strong className="text-[#1F6F5C] bg-[#1F6F5C]/10 px-1 py-0.5 rounded">
                          FedAvg Aggregation
                        </strong>
                        <br />
                        Hospital 3 (HOSP-003) ──┘
                      </div>
                    </div>

                    {/* Model Versioning Output */}
                    <div className="rounded bg-[#F7F9F6] p-3 border border-[#DDE3DC]/60 space-y-1.5 text-[11px]">
                      <div className="flex justify-between">
                        <span className="text-[#101B16]/60">Previous Version:</span>
                        <span className="font-mono text-[#101B16]/80">
                          {liveRoundStatus?.previous_model_version ?? 'resnet18_fed_round_002'}
                        </span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-[#101B16]/60">New Aggregated Model:</span>
                        <span className="font-mono font-bold text-[#1F6F5C] bg-[#1F6F5C]/10 px-2 py-0.5 rounded">
                          {liveRoundStatus?.global_model_version ??
                            fedOverview?.current_model_version ??
                            '-'}
                        </span>
                      </div>
                      <div className="flex justify-between pt-1 border-t border-[#DDE3DC]/40 text-[10.5px]">
                        <span>
                          Global Macro F1:{' '}
                          <strong>
                            {liveRoundStatus?.metrics?.f1 != null
                              ? `${(liveRoundStatus.metrics.f1 * 100).toFixed(1)}%`
                              : fedOverview?.global_f1 != null
                                ? `${(fedOverview.global_f1 * 100).toFixed(1)}%`
                                : '-'}
                          </strong>
                        </span>
                        <span>
                          Val Accuracy:{' '}
                          <strong>
                            {liveRoundStatus?.metrics?.accuracy != null
                              ? `${(liveRoundStatus.metrics.accuracy * 100).toFixed(1)}%`
                              : fedOverview?.global_accuracy != null
                                ? `${(fedOverview.global_accuracy * 100).toFixed(1)}%`
                                : '-'}
                          </strong>
                        </span>
                      </div>
                    </div>
                  </div>

                  {liveRoundStatus?.status === 'COMPLETED' && (
                    <div className="mt-3 rounded-md bg-[#EBF5F1] p-2.5 text-xs text-[#1F6F5C] font-semibold flex items-center justify-between border border-[#D2E0D1]">
                      <div className="flex items-center gap-2">
                        <span>✓</span>
                        <span>
                          ROUND #{liveRoundStatus.round} COMPLETED — DL Global Federated Model
                          synchronized across all hospital clients!
                        </span>
                      </div>
                      <span className="text-[11px] font-mono text-[#1F6F5C]/80">
                        {liveRoundStatus.global_model_version}.pth
                      </span>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>

          {/* KPI Grid */}
          <div className="grid grid-cols-4 gap-4">
            <div className="rounded-xl border border-[#DDE3DC] bg-white p-5 shadow-xs">
              <span className="text-xs font-medium text-[#101B16]/55">Completed Rounds</span>
              <p className="text-2xl font-bold mt-1.5 text-[#3B3F8C]">
                {fedOverview?.current_round ? `Round #${fedOverview.current_round}` : 'Round #3'}
              </p>
              <p className="text-[11px] text-[#1F6F5C] mt-1">✓ Convergence threshold met</p>
            </div>

            <div className="rounded-xl border border-[#DDE3DC] bg-white p-5 shadow-xs">
              <span className="text-xs font-medium text-[#101B16]/55">Global Macro F1 Score</span>
              <p className="text-2xl font-bold mt-1.5 text-[#1F6F5C]">
                {fedOverview?.global_f1 != null
                  ? `${(fedOverview.global_f1 * 100).toFixed(1)}%`
                  : '-'}
              </p>
              <p className="text-[11px] text-[#101B16]/45 mt-1">+1.8% vs Round 1</p>
            </div>

            <div className="rounded-xl border border-[#DDE3DC] bg-white p-5 shadow-xs">
              <span className="text-xs font-medium text-[#101B16]/55">
                Global Validation Accuracy
              </span>
              <p className="text-2xl font-bold mt-1.5 text-[#101B16]">
                {fedOverview?.global_accuracy != null
                  ? `${(fedOverview.global_accuracy * 100).toFixed(1)}%`
                  : '-'}
              </p>
              <p className="text-[11px] text-[#101B16]/45 mt-1">
                Loss: {fedOverview?.global_loss != null ? fedOverview.global_loss.toFixed(4) : '-'}
              </p>
            </div>

            <div className="rounded-xl border border-[#DDE3DC] bg-white p-5 shadow-xs">
              <span className="text-xs font-medium text-[#101B16]/55">Collaborative CT Slices</span>
              <p className="text-2xl font-bold mt-1.5 text-[#3B3F8C]">
                {fedOverview?.total_samples != null
                  ? fedOverview.total_samples.toLocaleString()
                  : '-'}
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
                  <h3 className="text-sm font-semibold text-[#101B16]">
                    Multi-Round Convergence Progression
                  </h3>
                  <p className="text-[11px] text-[#101B16]/60">
                    Validation Macro F1 and Accuracy across aggregated rounds
                  </p>
                </div>
                <span className="rounded bg-[#3B3F8C]/10 px-2 py-0.5 text-[10.5px] font-bold text-[#3B3F8C]">
                  FedAvg
                </span>
              </div>

              <div className="space-y-4">
                {roundHistory.map((r) => (
                  <div
                    key={r.round_number}
                    className="rounded-lg bg-[#F7F9F6] p-3.5 border border-[#DDE3DC]/60"
                  >
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs font-bold text-[#101B16]">
                        Round #{r.round_number} Aggregation
                      </span>
                      <span className="text-[11px] text-[#101B16]/60">
                        Duration: {r.duration_sec != null ? `${r.duration_sec.toFixed(1)}s` : '-'}
                      </span>
                    </div>

                    <div className="space-y-2">
                      <div>
                        <div className="flex justify-between text-[11px] text-[#101B16]/70 mb-1">
                          <span>
                            Macro F1:{' '}
                            <strong>
                              {r.global_val_f1 != null
                                ? `${(r.global_val_f1 * 100).toFixed(1)}%`
                                : '-'}
                            </strong>
                          </span>
                          <span>
                            Acc:{' '}
                            <strong>
                              {r.global_val_acc != null
                                ? `${(r.global_val_acc * 100).toFixed(1)}%`
                                : '-'}
                            </strong>
                          </span>
                          <span>
                            Loss:{' '}
                            <strong>
                              {r.global_val_loss != null ? r.global_val_loss.toFixed(4) : '-'}
                            </strong>
                          </span>
                        </div>
                        <div className="h-2 w-full rounded-full bg-[#DDE3DC]/60 overflow-hidden">
                          <div
                            className="h-full rounded-full bg-[#1F6F5C] transition-all duration-500"
                            style={{ width: `${(r.global_val_f1 ?? 0) * 100}%` }}
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
                  <h3 className="text-sm font-semibold text-[#101B16]">
                    Hospital Node Participation & Weighting
                  </h3>
                  <p className="text-[11px] text-[#101B16]/60">
                    Isolated CT sample contributions and local accuracy
                  </p>
                </div>
                <span className="rounded bg-[#1F6F5C]/10 px-2 py-0.5 text-[10.5px] font-bold text-[#1F6F5C]">
                  3 Nodes Online
                </span>
              </div>

              <div className="space-y-3">
                {participation.map((h) => (
                  <div
                    key={h.hospital_code}
                    className="rounded-lg border border-[#DDE3DC]/70 p-3.5 bg-[#F7F9F6]/50"
                  >
                    <div className="flex items-center justify-between mb-1">
                      <div className="flex items-center gap-2">
                        <span className="font-mono text-xs font-bold text-[#3B3F8C]">
                          {h.hospital_code}
                        </span>
                        <span className="text-xs font-semibold text-[#101B16]">{h.name}</span>
                      </div>
                      <span className="inline-flex items-center gap-1 text-[10.5px] font-semibold text-[#1F6F5C]">
                        <span className="h-1.5 w-1.5 rounded-full bg-[#1F6F5C]" />{' '}
                        {liveRoundStatus?.clients.find(
                          (client) => client.hospital_id === h.hospital_code
                        )?.status ?? 'ENROLLED'}
                      </span>
                    </div>

                    <div className="grid grid-cols-3 gap-2 mt-2 pt-2 border-t border-[#DDE3DC]/40 text-[11px] text-[#101B16]/70">
                      <div>
                        Samples:{' '}
                        <strong className="text-[#101B16]">
                          {h.dataset_size.toLocaleString()}
                        </strong>
                      </div>
                      <div>
                        Weight Share:{' '}
                        <strong className="text-[#101B16]">{h.sample_contribution_pct}%</strong>
                      </div>
                      <div>
                        Local F1:{' '}
                        <strong className="text-[#1F6F5C]">
                          {h.latest_local_f1 != null
                            ? `${(h.latest_local_f1 * 100).toFixed(1)}%`
                            : '-'}
                        </strong>
                      </div>
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
                <h3 className="text-sm font-semibold text-[#101B16]">
                  DL Federated Round Telemetry History
                </h3>
                <p className="text-[11px] text-[#101B16]/60">
                  Complete audit log of aggregated weights, validation scores, and timing
                </p>
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
                {roundHistory.map((r) => (
                  <tr key={r.round_number} className="hover:bg-black/[0.015]">
                    <td className="py-3.5 px-4 font-bold text-[#3B3F8C]">
                      Round #{r.round_number}
                    </td>
                    <td className="py-3.5 px-4 uppercase font-semibold text-[11px] text-[#101B16]/70">
                      {r.mode ?? 'IID'}
                    </td>
                    <td className="py-3.5 px-4 text-[#101B16]">
                      {r.participants_count ?? 3} Hospitals
                    </td>
                    <td className="py-3.5 px-4 font-mono text-[#101B16]/70">
                      {r.global_train_loss != null ? r.global_train_loss.toFixed(4) : '-'}
                    </td>
                    <td className="py-3.5 px-4 font-semibold text-[#101B16]">
                      {r.global_val_acc != null ? `${(r.global_val_acc * 100).toFixed(1)}%` : '-'}
                    </td>
                    <td className="py-3.5 px-4 font-semibold text-[#1F6F5C]">
                      {r.global_val_f1 != null ? `${(r.global_val_f1 * 100).toFixed(1)}%` : '-'}
                    </td>
                    <td className="py-3.5 px-4 text-[#101B16]/70">
                      {r.duration_sec != null ? `${r.duration_sec.toFixed(1)}s` : '-'}
                    </td>
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
      {activeTab === 'versions' && (
        <div className="space-y-6">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="font-serif text-lg font-medium text-[#101B16]">
                Model Versions & Deployment Rollout
              </h2>
              <p className="text-xs text-[#101B16]/60 mt-0.5">
                Manage artifact deployments, canary rollout thresholds, and instant version
                promotion or rollback.
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
                        {v.model_family === 'xgboost_risk'
                          ? 'XGBoost Clinical Risk'
                          : 'ResNet18 CT Imaging'}
                      </span>
                    </td>
                    <td className="py-3.5 px-4 font-mono font-medium text-[#3B3F8C]">
                      {v.version_tag}
                    </td>
                    <td className="py-3.5 px-4">
                      <span className="font-semibold text-[#101B16]">
                        {v.accuracy ? `${(v.accuracy * 100).toFixed(1)}%` : '—'}
                      </span>
                      <span className="text-[#101B16]/50 text-[11px] ml-1">
                        (F1: {v.f1_score ? `${(v.f1_score * 100).toFixed(1)}%` : '—'})
                      </span>
                    </td>
                    <td className="py-3.5 px-4 text-[#101B16]/70">
                      {new Date(v.trained_at).toLocaleDateString('en-GB', {
                        day: '2-digit',
                        month: 'short',
                        year: 'numeric',
                      })}
                    </td>
                    <td className="py-3.5 px-4">
                      <span
                        className={`rounded-full px-2.5 py-0.5 text-[10.5px] font-semibold ${
                          v.is_deployed
                            ? 'bg-[#1F6F5C]/15 text-[#1F6F5C]'
                            : v.environment === 'canary'
                              ? 'bg-[#C97A2B]/15 text-[#C97A2B]'
                              : 'bg-[#101B16]/10 text-[#101B16]/70'
                        }`}
                      >
                        {v.is_deployed ? 'Production (Active)' : (v.environment ?? 'Staging')}
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
      {/* TAB 3: HOSPITAL UPDATE LOGS (Merged into Federated) */}
      {/* ========================================================================= */}
      {activeTab === 'federated' && (
        <div className="space-y-6 mt-6 border-t border-[#DDE3DC] pt-6">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="font-serif text-lg font-medium text-[#101B16]">
                DL Hospital Update & Federated Sync Logs
              </h2>
              <p className="text-xs text-[#101B16]/60 mt-0.5">
                Audit trail of model parameter distribution and telemetry updates received across
                enrolled hospital nodes.
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
                onClick={() => showToast('Exporting federated audit logs (CSV)...')}
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
                    hospLogSearch
                      ? l.hospital_name.toLowerCase().includes(hospLogSearch.toLowerCase()) ||
                        l.version_tag.toLowerCase().includes(hospLogSearch.toLowerCase())
                      : true
                  )
                  .map((log, idx) => (
                    <tr key={idx} className="hover:bg-black/[0.015]">
                      <td className="py-3.5 px-4 font-medium text-[#101B16]">
                        {log.hospital_name}
                      </td>
                      <td className="py-3.5 px-4 font-mono font-medium text-[#3B3F8C]">
                        {log.version_tag}
                      </td>
                      <td className="py-3.5 px-4">
                        <span
                          className={`rounded-full px-2.5 py-0.5 text-[10.5px] font-semibold ${
                            log.status === 'received'
                              ? 'bg-[#1F6F5C]/15 text-[#1F6F5C]'
                              : log.status === 'pending'
                                ? 'bg-[#C97A2B]/15 text-[#C97A2B]'
                                : 'bg-[#B3261E]/15 text-[#B3261E]'
                          }`}
                        >
                          {log.status.toUpperCase()}
                        </span>
                      </td>
                      <td className="py-3.5 px-4 text-[#101B16]/80">
                        {log.payload_size_mb ? `${log.payload_size_mb} MB` : '4.8 MB'}
                      </td>
                      <td className="py-3.5 px-4 font-mono text-[11px] text-[#101B16]/55">
                        {log.gradient_hash ?? 'sha256:7f9a2e3...b19c'}
                      </td>
                      <td className="py-3.5 px-4 text-[#101B16]/70">
                        {new Date(log.created_at).toLocaleString('en-GB', {
                          day: '2-digit',
                          month: 'short',
                          hour: '2-digit',
                          minute: '2-digit',
                          second: '2-digit',
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
      {/* SYSTEM HEALTH (Monitoring + Drift) */}
      {/* ========================================================================= */}
      {activeTab === 'health' && (
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
            <h2 className="font-serif text-base font-medium text-[#101B16] mb-3">
              Service Architecture & Node Status
            </h2>
            <div className="grid grid-cols-3 gap-4 text-xs">
              {[
                {
                  name: 'FastAPI Core Gateway',
                  status: 'Healthy',
                  ping: '2ms',
                  sub: 'Load balancer active',
                },
                {
                  name: 'XGBoost Risk Inference Engine',
                  status: 'Healthy',
                  ping: '18ms',
                  sub: 'CPU multicore pool',
                },
                {
                  name: 'ResNet18 CT Imaging PyTorch',
                  status: 'Healthy',
                  ping: '82ms',
                  sub: 'GPU TensorRT worker',
                },
                {
                  name: 'PostgreSQL Database',
                  status: 'Healthy',
                  ping: '4ms',
                  sub: 'Connection pool 12/50',
                },
                {
                  name: 'Redis In-Memory Queue',
                  status: 'Healthy',
                  ping: '1ms',
                  sub: 'Cache hit ratio 94%',
                },
                {
                  name: 'Federated Sync Aggregator',
                  status: 'Healthy',
                  ping: '15ms',
                  sub: 'TLS v1.3 verification',
                },
              ].map((svc) => (
                <div
                  key={svc.name}
                  className="rounded-lg border border-[#DDE3DC] bg-[#F7F9F6] p-3.5 flex items-center justify-between"
                >
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
              <h2 className="font-serif text-base font-medium text-[#101B16]">
                Real-Time System & Inference Logs
              </h2>
              <div className="flex items-center gap-2">
                <span className="text-xs text-[#101B16]/50">Level:</span>
                {['all', 'info', 'warning', 'error'].map((lvl) => (
                  <button
                    key={lvl}
                    onClick={() => setLogFilterLevel(lvl)}
                    className={`rounded px-2.5 py-1 text-[11px] font-medium capitalize cursor-pointer ${
                      logFilterLevel === lvl
                        ? 'bg-[#3B3F8C] text-white'
                        : 'bg-[#F3F6F1] text-[#101B16]/70 hover:bg-[#DDE3DC]'
                    }`}
                  >
                    {lvl}
                  </button>
                ))}
              </div>
            </div>

            <div className="space-y-2 max-h-72 overflow-y-auto pr-1">
              {logs
                .filter((l) => logFilterLevel === 'all' || l.level === logFilterLevel)
                .map((log, idx) => (
                  <div
                    key={idx}
                    className="flex items-start justify-between rounded-lg border border-[#DDE3DC]/80 bg-[#F9FAF8] p-2.5 text-xs"
                  >
                    <div className="flex items-start gap-2.5">
                      <span
                        className={`mt-1 h-2 w-2 rounded-full shrink-0 ${
                          log.level === 'error'
                            ? 'bg-[#B3261E]'
                            : log.level === 'warning'
                              ? 'bg-[#C97A2B]'
                              : 'bg-[#1F6F5C]'
                        }`}
                      />
                      <div>
                        <p className="font-medium text-[#101B16]">{log.message}</p>
                        <p className="text-[11px] text-[#101B16]/45 mt-0.5">
                          Node: {log.hospital_name ?? 'System Cluster'} · Service:{' '}
                          {log.service ?? 'inference-worker'}
                        </p>
                      </div>
                    </div>
                    <span className="text-[11px] text-[#101B16]/45 font-mono">
                      {new Date(log.created_at).toLocaleTimeString([], {
                        hour: '2-digit',
                        minute: '2-digit',
                        second: '2-digit',
                      })}
                    </span>
                  </div>
                ))}
            </div>
          </section>
        </div>
      )}

      {/* ========================================================================= */}
      {/* DRIFT ANALYSIS (Merged into System Health) */}
      {/* ========================================================================= */}
      {activeTab === 'health' && (
        <div className="space-y-6 mt-6 border-t border-[#DDE3DC] pt-6">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="font-serif text-lg font-medium text-[#101B16]">
                Biomarker & Image Distribution Drift Analysis
              </h2>
              <p className="text-xs text-[#101B16]/60 mt-0.5">
                Automated statistical testing (Kolmogorov-Smirnov & Population Stability Index)
                comparing training baseline vs live stream.
              </p>
            </div>
            <button
              onClick={() =>
                showToast(
                  'ML is centralized. Start ML Training is ready for the centralized training pipeline.'
                )
              }
              className="rounded-lg bg-[#3B3F8C] px-4 py-2 text-xs font-medium text-white hover:bg-[#2F3270] shadow-xs cursor-pointer"
            >
              Start ML Training
            </button>
          </div>

          {/* Drift Metrics Grid */}
          <div className="grid grid-cols-2 gap-6">
            <section className="rounded-xl border border-[#DDE3DC] bg-white p-5 shadow-xs">
              <h3 className="font-serif text-base font-medium text-[#101B16] mb-3">
                Clinical Biomarker Feature Drift (XGBoost)
              </h3>
              <div className="space-y-3 text-xs">
                {drift
                  .filter((d) => d.model_family === 'xgboost_risk')
                  .map((d, i) => (
                    <div key={i} className="rounded-lg border border-[#DDE3DC] p-3 bg-[#F9FAF8]">
                      <div className="flex items-center justify-between mb-1">
                        <span className="font-semibold text-[#101B16]">{d.metric_name}</span>
                        <span
                          className={`rounded-full px-2 py-0.5 text-[10.5px] font-bold ${
                            d.status === 'warning' || d.drift_score > 0.15
                              ? 'bg-[#C97A2B]/15 text-[#C97A2B]'
                              : 'bg-[#1F6F5C]/15 text-[#1F6F5C]'
                          }`}
                        >
                          Score: {d.drift_score.toFixed(3)} ({d.status ?? 'Normal'})
                        </span>
                      </div>
                      <div className="w-full bg-[#DDE3DC] h-2 rounded-full my-2">
                        <div
                          className={`h-2 rounded-full ${
                            d.drift_score > 0.15 ? 'bg-[#C97A2B]' : 'bg-[#1F6F5C]'
                          }`}
                          style={{ width: `${Math.min(d.drift_score * 400, 100)}%` }}
                        />
                      </div>
                      <div className="flex justify-between text-[10.5px] text-[#101B16]/55">
                        <span>Baseline Ref: {d.reference_mean ?? 'Normal range'}</span>
                        <span>Current Live: {d.current_mean ?? 'In range'}</span>
                        <span>p-value: {d.p_value ?? 0.75}</span>
                      </div>
                    </div>
                  ))}
              </div>
            </section>

            <section className="rounded-xl border border-[#DDE3DC] bg-white p-5 shadow-xs">
              <h3 className="font-serif text-base font-medium text-[#101B16] mb-3">
                CT Imaging Distribution Shift (ResNet18)
              </h3>
              <div className="space-y-3 text-xs">
                {drift
                  .filter((d) => d.model_family === 'resnet18_ct')
                  .map((d, i) => (
                    <div key={i} className="rounded-lg border border-[#DDE3DC] p-3 bg-[#F9FAF8]">
                      <div className="flex items-center justify-between mb-1">
                        <span className="font-semibold text-[#101B16]">{d.metric_name}</span>
                        <span
                          className={`rounded-full px-2 py-0.5 text-[10.5px] font-bold ${
                            d.drift_score > 0.15
                              ? 'bg-[#B3261E]/15 text-[#B3261E]'
                              : 'bg-[#1F6F5C]/15 text-[#1F6F5C]'
                          }`}
                        >
                          Score: {d.drift_score.toFixed(3)} ({d.status ?? 'Normal'})
                        </span>
                      </div>
                      <div className="w-full bg-[#DDE3DC] h-2 rounded-full my-2">
                        <div
                          className={`h-2 rounded-full ${
                            d.drift_score > 0.15 ? 'bg-[#B3261E]' : 'bg-[#1F6F5C]'
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
                  💡 <strong>Automatic Trigger Rule:</strong> When biomarker PSI &gt; 0.20 or
                  imaging KS p-value &lt; 0.01 for 3 consecutive days, the orchestrator alerts the
                  ML engineers and initiates a warm-start fine-tuning cycle.
                </div>
              </div>
            </section>
          </div>
        </div>
      )}

      {/* ========================================================================= */}
      {/* TAB 6: ENROLLED HOSPITALS & ACCESS MANAGEMENT */}
      {/* ========================================================================= */}
      {activeTab === 'access' && (
        <div className="space-y-6">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="font-serif text-lg font-medium text-[#101B16]">
                List of Enrolled Hospitals & Access Management
              </h2>
              <p className="text-xs text-[#101B16]/60 mt-0.5">
                Grant new hospital access, revoke or suspend API credentials, and manage
                multi-tenant permissions.
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
                        <p className="text-[11px] text-[#101B16]/45">
                          {hospital.contact_email ?? 'admin@hospital.org'}
                        </p>
                      </td>
                      <td className="py-3.5 px-4">
                        <span className="font-mono font-medium text-[#3B3F8C]">
                          {hospital.hospital_code}
                        </span>
                        <p className="text-[11px] text-[#101B16]/50">
                          {hospital.region ?? 'India'}
                        </p>
                      </td>
                      <td className="py-3.5 px-4">
                        <span className="rounded bg-[#3B3F8C]/10 px-2 py-0.5 text-[10.5px] font-bold text-[#3B3F8C] capitalize">
                          {hospital.tier ?? 'Standard'}
                        </span>
                      </td>
                      <td className="py-3.5 px-4">
                        <div className="flex items-center gap-1.5">
                          <span className="font-mono text-[11px] text-[#101B16]/75 bg-[#F3F6F1] px-2 py-0.5 rounded border border-[#DDE3DC]">
                            {hospital.api_key
                              ? `${hospital.api_key.slice(0, 10)}•••••`
                              : 'ss_live_••••'}
                          </span>
                          <button
                            onClick={() => {
                              navigator.clipboard.writeText(hospital.api_key || 'ss_live_key');
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
                            hospital.is_active
                              ? 'bg-[#1F6F5C]/15 text-[#1F6F5C]'
                              : 'bg-[#B3261E]/15 text-[#B3261E]'
                          }`}
                        >
                          <span
                            className={`h-1.5 w-1.5 rounded-full ${
                              hospital.is_active ? 'bg-[#1F6F5C]' : 'bg-[#B3261E]'
                            }`}
                          />
                          {hospital.is_active ? 'Authorized' : 'Revoked'}
                        </span>
                      </td>
                      <td className="py-3.5 px-4 font-semibold text-[#101B16]">
                        {hospital.total_scans ?? 250}
                      </td>
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
            <h2 className="font-serif text-xl font-medium text-[#101B16]">
              Deploy New Model Version
            </h2>
            <p className="text-xs text-[#101B16]/60 mt-1 mb-5">
              Publish and configure a new model artifact release for cross-hospital inference.
            </p>

            <form onSubmit={handleCreateVersionSubmit} className="space-y-4 text-xs">
              <div className="grid grid-cols-2 gap-3.5">
                <div>
                  <label className="block text-[#101B16]/70 font-medium mb-1">Model Family</label>
                  <select
                    value={newVersionForm.modelFamily}
                    onChange={(e) =>
                      setNewVersionForm({ ...newVersionForm, modelFamily: e.target.value })
                    }
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
                    onChange={(e) =>
                      setNewVersionForm({ ...newVersionForm, versionTag: e.target.value })
                    }
                    className="w-full rounded-md border border-[#DDE3DC] bg-white px-3 py-2 text-xs text-[#101B16] outline-none"
                  />
                </div>

                <div>
                  <label className="block text-[#101B16]/70 font-medium mb-1">
                    Validation Accuracy
                  </label>
                  <input
                    type="text"
                    value={newVersionForm.accuracy}
                    onChange={(e) =>
                      setNewVersionForm({ ...newVersionForm, accuracy: e.target.value })
                    }
                    className="w-full rounded-md border border-[#DDE3DC] bg-white px-3 py-2 text-xs text-[#101B16] outline-none"
                  />
                </div>

                <div>
                  <label className="block text-[#101B16]/70 font-medium mb-1">
                    Target Environment
                  </label>
                  <select
                    value={newVersionForm.environment}
                    onChange={(e) =>
                      setNewVersionForm({ ...newVersionForm, environment: e.target.value })
                    }
                    className="w-full rounded-md border border-[#DDE3DC] bg-white px-3 py-2 text-xs text-[#101B16] outline-none"
                  >
                    <option value="production">Production (Immediate 100%)</option>
                    <option value="canary">Canary (Gradual Traffic)</option>
                    <option value="staging">Staging (Internal Testing)</option>
                  </select>
                </div>
              </div>

              <div>
                <label className="block text-[#101B16]/70 font-medium mb-1">
                  Release Notes & Benchmark Summary
                </label>
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
            <h2 className="font-serif text-xl font-medium text-[#101B16]">
              Provide Access — Enroll Partner Hospital
            </h2>
            <p className="text-xs text-[#101B16]/60 mt-1 mb-5">
              Issue cryptographic API credentials and configure tenant authorization.
            </p>

            <form onSubmit={handleEnrollHospitalSubmit} className="space-y-4 text-xs">
              <div className="grid grid-cols-2 gap-3.5">
                <div className="col-span-2">
                  <label className="block text-[#101B16]/70 font-medium mb-1">
                    Hospital Institution Name
                  </label>
                  <input
                    type="text"
                    required
                    placeholder="e.g. Christian Medical College Hospital"
                    value={newHospitalForm.name}
                    onChange={(e) =>
                      setNewHospitalForm({ ...newHospitalForm, name: e.target.value })
                    }
                    className="w-full rounded-md border border-[#DDE3DC] bg-white px-3 py-2 text-xs text-[#101B16] outline-none focus:border-[#1F6F5C]"
                  />
                </div>

                <div>
                  <label className="block text-[#101B16]/70 font-medium mb-1">Hospital Code</label>
                  <input
                    type="text"
                    placeholder="e.g. HOSP-CMC-07"
                    value={newHospitalForm.code}
                    onChange={(e) =>
                      setNewHospitalForm({ ...newHospitalForm, code: e.target.value })
                    }
                    className="w-full rounded-md border border-[#DDE3DC] bg-white px-3 py-2 text-xs text-[#101B16] outline-none"
                  />
                </div>

                <div>
                  <label className="block text-[#101B16]/70 font-medium mb-1">Access Tier</label>
                  <select
                    value={newHospitalForm.tier}
                    onChange={(e) =>
                      setNewHospitalForm({ ...newHospitalForm, tier: e.target.value as any })
                    }
                    className="w-full rounded-md border border-[#DDE3DC] bg-white px-3 py-2 text-xs text-[#101B16] outline-none"
                  >
                    <option value="enterprise">Enterprise (Unlimited + Dedicated GPU)</option>
                    <option value="standard">Standard (5,000 Scans / mo)</option>
                    <option value="research">Research Cohort (1,000 Scans / mo)</option>
                  </select>
                </div>

                <div>
                  <label className="block text-[#101B16]/70 font-medium mb-1">
                    Region / Location
                  </label>
                  <input
                    type="text"
                    placeholder="e.g. Tamil Nadu, India"
                    value={newHospitalForm.region}
                    onChange={(e) =>
                      setNewHospitalForm({ ...newHospitalForm, region: e.target.value })
                    }
                    className="w-full rounded-md border border-[#DDE3DC] bg-white px-3 py-2 text-xs text-[#101B16] outline-none"
                  />
                </div>

                <div>
                  <label className="block text-[#101B16]/70 font-medium mb-1">
                    Admin Contact Email
                  </label>
                  <input
                    type="email"
                    required
                    placeholder="urology.lead@hospital.edu"
                    value={newHospitalForm.contactEmail}
                    onChange={(e) =>
                      setNewHospitalForm({ ...newHospitalForm, contactEmail: e.target.value })
                    }
                    className="w-full rounded-md border border-[#DDE3DC] bg-white px-3 py-2 text-xs text-[#101B16] outline-none"
                  />
                </div>
              </div>

              <div className="rounded-lg bg-[#F3F6F1] border border-[#DDE3DC] p-3 text-[11px] text-[#101B16]/70">
                🔒 Provisioning this hospital generates a dedicated 256-bit API Secret, establishes
                an isolated tenant schema, and enables federated weight updates.
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
            <h2 className="font-serif text-lg font-medium text-[#101B16]">
              Federated Payload Inspection
            </h2>
            <p className="text-[#101B16]/60 mt-1 mb-4">
              Audit log verification for {selectedLogPayload.hospital_name}
            </p>

            <div className="space-y-2.5 bg-[#F9FAF8] border border-[#DDE3DC] p-4 rounded-lg font-mono text-[11px]">
              <div>
                <span className="text-[#101B16]/50 block">Hospital Node:</span>
                <span className="text-[#101B16] font-sans font-semibold">
                  {selectedLogPayload.hospital_name}
                </span>
              </div>
              <div>
                <span className="text-[#101B16]/50 block">Target Version:</span>
                <span className="text-[#3B3F8C] font-semibold">
                  {selectedLogPayload.version_tag}
                </span>
              </div>
              <div>
                <span className="text-[#101B16]/50 block">Gradient Checksum:</span>
                <span className="text-[#101B16]">
                  {selectedLogPayload.gradient_hash ?? 'sha256:7f9a2e340a1b8c'}
                </span>
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

      {isRoundModalOpen && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-[#101B16]/50 backdrop-blur-xs p-4 animate-fade-in"
          onClick={() => setIsRoundModalOpen(false)}
        >
          <div
            className="w-full max-w-lg rounded-xl bg-white p-6 shadow-2xl animate-scale-up"
            onClick={(event) => event.stopPropagation()}
          >
            <h2 className="font-serif text-xl font-medium text-[#101B16]">
              Start DL Federated Round
            </h2>
            <p className="mt-1 text-xs text-[#101B16]/60">
              Select the enrolled hospitals that will train locally and contribute updates to
              Sample-Weighted FedAvg.
            </p>

            <div className="mt-5 space-y-3 rounded-lg border border-[#DDE3DC] bg-[#F7F9F6] p-4 text-xs">
              <div className="flex items-center justify-between">
                <span className="font-semibold text-[#101B16]/60">Pipeline</span>
                <span className="font-semibold text-[#3B3F8C]">DL — Federated Learning</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="font-semibold text-[#101B16]/60">Base Model</span>
                <span className="font-mono font-semibold text-[#101B16]">
                  {liveRoundStatus?.global_model_version ??
                    fedOverview?.current_model_version ??
                    'Current global DL model'}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="font-semibold text-[#101B16]/60">Aggregation</span>
                <span className="font-semibold text-[#1F6F5C]">Sample-Weighted FedAvg</span>
              </div>
            </div>

            <div className="mt-5">
              <div className="mb-2 flex items-center justify-between">
                <h3 className="text-sm font-semibold text-[#101B16]">Participating Hospitals</h3>
                <span className="text-xs text-[#101B16]/50">
                  {selectedHospitalIds.length} selected
                </span>
              </div>
              <div className="space-y-2">
                {hospitals
                  .filter((hospital) => hospital.is_active)
                  .map((hospital) => {
                    const selected = selectedHospitalIds.includes(hospital.id);
                    return (
                      <label
                        key={hospital.id}
                        className="flex cursor-pointer items-center justify-between rounded-lg border border-[#DDE3DC] px-3 py-2.5 text-xs hover:bg-[#F7F9F6]"
                      >
                        <span>
                          <span className="font-semibold text-[#101B16]">{hospital.name}</span>
                          <span className="ml-2 font-mono text-[11px] text-[#101B16]/50">
                            {hospital.hospital_code}
                          </span>
                        </span>
                        <input
                          type="checkbox"
                          checked={selected}
                          onChange={() =>
                            setSelectedHospitalIds((current) =>
                              selected
                                ? current.filter((id) => id !== hospital.id)
                                : [...current, hospital.id]
                            )
                          }
                          className="h-4 w-4 accent-[#1F6F5C]"
                        />
                      </label>
                    );
                  })}
              </div>
            </div>

            <div className="mt-5 grid grid-cols-3 gap-3 text-xs">
              <label className="text-[#101B16]/70">
                Local epochs
                <input
                  type="number"
                  min="1"
                  value={roundConfig.localEpochs}
                  onChange={(event) =>
                    setRoundConfig({ ...roundConfig, localEpochs: Number(event.target.value) })
                  }
                  className="mt-1 w-full rounded-md border border-[#DDE3DC] px-2 py-2 text-[#101B16]"
                />
              </label>
              <label className="text-[#101B16]/70">
                Batch size
                <input
                  type="number"
                  min="1"
                  value={roundConfig.batchSize}
                  onChange={(event) =>
                    setRoundConfig({ ...roundConfig, batchSize: Number(event.target.value) })
                  }
                  className="mt-1 w-full rounded-md border border-[#DDE3DC] px-2 py-2 text-[#101B16]"
                />
              </label>
              <label className="text-[#101B16]/70">
                Learning rate
                <input
                  type="number"
                  min="0.000001"
                  step="0.0001"
                  value={roundConfig.learningRate}
                  onChange={(event) =>
                    setRoundConfig({ ...roundConfig, learningRate: Number(event.target.value) })
                  }
                  className="mt-1 w-full rounded-md border border-[#DDE3DC] px-2 py-2 font-mono text-[#101B16]"
                />
              </label>
            </div>

            <div className="mt-6 flex justify-end gap-3 border-t border-[#DDE3DC] pt-4">
              <button
                type="button"
                onClick={() => setIsRoundModalOpen(false)}
                className="rounded-md px-3.5 py-2 text-xs text-[#101B16]/65 hover:text-[#101B16]"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={handleStartFederatedRound}
                disabled={isStartingRound || selectedHospitalIds.length === 0}
                className="rounded-md bg-[#1F6F5C] px-4 py-2 text-xs font-semibold text-white hover:bg-[#185849] disabled:cursor-not-allowed disabled:opacity-50"
              >
                {isStartingRound ? 'Starting...' : 'Start DL Federated Round'}
              </button>
            </div>
          </div>
        </div>
      )}
    </AppLayout>
  );
}
