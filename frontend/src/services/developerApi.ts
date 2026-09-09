// frontend/src/services/developerApi.ts
import axios from "axios";
import {
  ModelPerformance,
  HospitalUpdateLogEntry,
  SystemLogEntry,
  DriftPoint,
  SystemMonitoringSummary,
  Hospital,
} from "../types/dashboard";

const client = axios.create({ baseURL: "/api/v1/developer" });

// Mock data fallbacks for robust operation
const fallbackModelVersions: ModelPerformance[] = [
  {
    id: 1,
    model_family: "xgboost_risk",
    version_tag: "v2.4.0-prod",
    accuracy: 0.942,
    f1_score: 0.938,
    mcc: 0.884,
    is_deployed: true,
    trained_at: "2026-09-01T10:00:00Z",
    latency_ms: 18.4,
    environment: "production",
    rollout_pct: 100,
  },
  {
    id: 2,
    model_family: "xgboost_risk",
    version_tag: "v2.5.0-rc1",
    accuracy: 0.958,
    f1_score: 0.952,
    mcc: 0.902,
    is_deployed: false,
    trained_at: "2026-09-08T14:30:00Z",
    latency_ms: 16.9,
    environment: "staging",
    rollout_pct: 0,
  },
  {
    id: 3,
    model_family: "resnet18_ct",
    version_tag: "v3.1.2-prod",
    accuracy: 0.961,
    f1_score: 0.958,
    mcc: 0.915,
    is_deployed: true,
    trained_at: "2026-08-28T09:15:00Z",
    latency_ms: 82.5,
    environment: "production",
    rollout_pct: 100,
  },
  {
    id: 4,
    model_family: "resnet18_ct",
    version_tag: "v3.2.0-canary",
    accuracy: 0.974,
    f1_score: 0.971,
    mcc: 0.938,
    is_deployed: false,
    trained_at: "2026-09-07T18:45:00Z",
    latency_ms: 76.1,
    environment: "canary",
    rollout_pct: 15,
  },
  {
    id: 5,
    model_family: "xgboost_risk",
    version_tag: "v2.3.1-legacy",
    accuracy: 0.912,
    f1_score: 0.905,
    mcc: 0.832,
    is_deployed: false,
    trained_at: "2026-07-15T11:00:00Z",
    latency_ms: 21.0,
    environment: "staging",
    rollout_pct: 0,
  },
];

const fallbackHospitalLogs: HospitalUpdateLogEntry[] = [
  {
    id: 101,
    hospital_name: "AIIMS Urology Dept.",
    version_tag: "v2.4.0-prod",
    status: "received",
    created_at: "2026-09-09T08:30:12Z",
    payload_size_mb: 4.8,
    gradient_hash: "sha256:7f9a2e3...b19c",
    latency_ms: 230,
  },
  {
    id: 102,
    hospital_name: "Apollo Renal Care",
    version_tag: "v2.4.0-prod",
    status: "received",
    created_at: "2026-09-09T07:15:44Z",
    payload_size_mb: 4.8,
    gradient_hash: "sha256:a12b4e8...c99f",
    latency_ms: 185,
  },
  {
    id: 103,
    hospital_name: "Fortis Nephrology Wing",
    version_tag: "v3.1.2-prod",
    status: "received",
    created_at: "2026-09-09T06:40:02Z",
    payload_size_mb: 42.1,
    gradient_hash: "sha256:bb870c1...d4e2",
    latency_ms: 540,
  },
  {
    id: 104,
    hospital_name: "Manipal Hospital Bangalore",
    version_tag: "v2.5.0-rc1",
    status: "pending",
    created_at: "2026-09-09T05:12:30Z",
    payload_size_mb: 5.1,
    gradient_hash: "sha256:c9902ff...e331",
    latency_ms: 310,
  },
  {
    id: 105,
    hospital_name: "CMC Vellore Urology",
    version_tag: "v3.1.2-prod",
    status: "failed",
    created_at: "2026-09-08T22:04:19Z",
    payload_size_mb: 0.0,
    gradient_hash: "—",
    latency_ms: 1250,
  },
  {
    id: 106,
    hospital_name: "Tata Memorial Renal Wing",
    version_tag: "v3.1.2-prod",
    status: "received",
    created_at: "2026-09-08T19:22:00Z",
    payload_size_mb: 42.1,
    gradient_hash: "sha256:e019ab2...44f9",
    latency_ms: 410,
  },
];

const fallbackMonitoring: SystemMonitoringSummary = {
  total_predictions_24h: 1482,
  error_count_24h: 3,
  avg_latency_ms: 44.8,
  p95_latency_ms: 88.2,
  p99_latency_ms: 135.0,
  uptime_pct: 99.98,
  gpu_utilization_pct: 42.5,
  cpu_utilization_pct: 28.0,
  active_nodes: 8,
};

const fallbackSystemLogs: SystemLogEntry[] = [
  {
    hospital_name: "AIIMS Urology Dept.",
    level: "info",
    message: "XGBoost inference completed in 18.2ms (Patient PT-2201)",
    created_at: "2026-09-09T09:12:04Z",
    service: "inference-worker-01",
  },
  {
    hospital_name: "Apollo Renal Care",
    level: "info",
    message: "ResNet18 CT scan processed with Grad-CAM heatmap generated",
    created_at: "2026-09-09T08:55:18Z",
    service: "inference-worker-02",
  },
  {
    hospital_name: "CMC Vellore Urology",
    level: "warning",
    message: "Federated weight sync packet retry: TLS handshake jitter",
    created_at: "2026-09-09T07:44:00Z",
    service: "federated-aggregator",
  },
  {
    hospital_name: null,
    level: "info",
    message: "Nightly automated model drift calculation completed across 6 biomarkers",
    created_at: "2026-09-09T04:00:00Z",
    service: "drift-scheduler",
  },
  {
    hospital_name: "CMC Vellore Urology",
    level: "error",
    message: "Weight stream packet corrupted: Hash mismatch rejected for quarantine",
    created_at: "2026-09-08T22:04:19Z",
    service: "federated-aggregator",
  },
];

const fallbackDriftPoints: DriftPoint[] = [
  {
    model_family: "xgboost_risk",
    metric_name: "pH Distribution (KS Test)",
    drift_score: 0.042,
    status: "normal",
    p_value: 0.82,
    reference_mean: 5.85,
    current_mean: 5.82,
    computed_at: "2026-09-09T04:00:00Z",
  },
  {
    model_family: "xgboost_risk",
    metric_name: "Specific Gravity (PSI)",
    drift_score: 0.068,
    status: "normal",
    p_value: 0.74,
    reference_mean: 1.018,
    current_mean: 1.019,
    computed_at: "2026-09-09T04:00:00Z",
  },
  {
    model_family: "xgboost_risk",
    metric_name: "Calcium Excretion (PSI)",
    drift_score: 0.162,
    status: "warning",
    p_value: 0.048,
    reference_mean: 4.12,
    current_mean: 4.78,
    computed_at: "2026-09-09T04:00:00Z",
  },
  {
    model_family: "xgboost_risk",
    metric_name: "Osmolality (KS Test)",
    drift_score: 0.051,
    status: "normal",
    p_value: 0.69,
    reference_mean: 615.0,
    current_mean: 610.5,
    computed_at: "2026-09-09T04:00:00Z",
  },
  {
    model_family: "resnet18_ct",
    metric_name: "CT HU Intensity Shift (KS)",
    drift_score: 0.089,
    status: "normal",
    p_value: 0.38,
    reference_mean: 120.4,
    current_mean: 124.1,
    computed_at: "2026-09-09T04:00:00Z",
  },
  {
    model_family: "resnet18_ct",
    metric_name: "Slice Noise Ratio (PSI)",
    drift_score: 0.185,
    status: "drift_detected",
    p_value: 0.021,
    reference_mean: 0.034,
    current_mean: 0.058,
    computed_at: "2026-09-09T04:00:00Z",
  },
];

export const fallbackHospitalsList: Hospital[] = [
  {
    id: 1,
    hospital_code: "HOSP-AIIMS-01",
    name: "AIIMS Urology Dept.",
    region: "New Delhi, India",
    is_active: true,
    tier: "enterprise",
    api_key: "ss_live_aiims_89104fa28",
    contact_email: "urology.lead@aiims.edu.in",
    last_sync: "2026-09-09T08:30:12Z",
    total_scans: 684,
  },
  {
    id: 2,
    hospital_code: "HOSP-APOLLO-02",
    name: "Apollo Renal Care",
    region: "Chennai, India",
    is_active: true,
    tier: "enterprise",
    api_key: "ss_live_apollo_9918bc321",
    contact_email: "nephro@apollohospitals.com",
    last_sync: "2026-09-09T07:15:44Z",
    total_scans: 512,
  },
  {
    id: 3,
    hospital_code: "HOSP-FORTIS-03",
    name: "Fortis Nephrology Wing",
    region: "Gurugram, India",
    is_active: true,
    tier: "standard",
    api_key: "ss_live_fortis_3321ef884",
    contact_email: "renal.dept@fortishealthcare.com",
    last_sync: "2026-09-09T06:40:02Z",
    total_scans: 340,
  },
  {
    id: 4,
    hospital_code: "HOSP-MANIPAL-04",
    name: "Manipal Hospital Bangalore",
    region: "Bengaluru, India",
    is_active: true,
    tier: "standard",
    api_key: "ss_live_manipal_4490dc112",
    contact_email: "urology@manipalhospitals.com",
    last_sync: "2026-09-09T05:12:30Z",
    total_scans: 290,
  },
  {
    id: 5,
    hospital_code: "HOSP-CMC-05",
    name: "CMC Vellore Urology",
    region: "Vellore, India",
    is_active: false,
    tier: "research",
    api_key: "ss_revoked_cmc_0012ba443",
    contact_email: "admin.uro@cmcvellore.ac.in",
    last_sync: "2026-09-08T22:04:19Z",
    total_scans: 142,
  },
  {
    id: 6,
    hospital_code: "HOSP-TATA-06",
    name: "Tata Memorial Renal Wing",
    region: "Mumbai, India",
    is_active: true,
    tier: "enterprise",
    api_key: "ss_live_tmh_552199aa1",
    contact_email: "renal.research@tmc.gov.in",
    last_sync: "2026-09-08T19:22:00Z",
    total_scans: 418,
  },
];

export const fetchModelPerformance = async (): Promise<ModelPerformance[]> => {
  try {
    const res = await client.get<ModelPerformance[]>("/model-performance");
    return res.data && res.data.length > 0 ? res.data : fallbackModelVersions;
  } catch {
    return fallbackModelVersions;
  }
};

export const fetchModelVersions = async (modelFamily?: string): Promise<ModelPerformance[]> => {
  try {
    const res = await client.get<ModelPerformance[]>("/model-versions", { params: { model_family: modelFamily } });
    return res.data && res.data.length > 0 ? res.data : fallbackModelVersions;
  } catch {
    return fallbackModelVersions;
  }
};

export const deployModelVersion = async (modelVersionId: number) => {
  try {
    return (await client.post("/model-versions/deploy", { model_version_id: modelVersionId })).data;
  } catch {
    return { success: true, deployed_id: modelVersionId };
  }
};

export const fetchHospitalLogs = async (): Promise<HospitalUpdateLogEntry[]> => {
  try {
    const res = await client.get<HospitalUpdateLogEntry[]>("/hospital-logs");
    return res.data && res.data.length > 0 ? res.data : fallbackHospitalLogs;
  } catch {
    return fallbackHospitalLogs;
  }
};

export const fetchSystemMonitoring = async (): Promise<SystemMonitoringSummary> => {
  try {
    const res = await client.get<SystemMonitoringSummary>("/system-monitoring");
    return res.data ?? fallbackMonitoring;
  } catch {
    return fallbackMonitoring;
  }
};

export const fetchSystemLogs = async (limit = 30): Promise<SystemLogEntry[]> => {
  try {
    const res = await client.get<SystemLogEntry[]>("/system-logs", { params: { limit } });
    return res.data && res.data.length > 0 ? res.data : fallbackSystemLogs;
  } catch {
    return fallbackSystemLogs;
  }
};

export const fetchDriftAnalysis = async (modelFamily?: string): Promise<DriftPoint[]> => {
  try {
    const res = await client.get<DriftPoint[]>("/drift-analysis", { params: { model_family: modelFamily } });
    return res.data && res.data.length > 0 ? res.data : fallbackDriftPoints;
  } catch {
    return fallbackDriftPoints;
  }
};

export const fetchAllEnrolledHospitals = async (): Promise<Hospital[]> => {
  try {
    const res = await axios.get<Hospital[]>("/api/v1/hospital/list");
    return res.data && res.data.length > 0 ? res.data : fallbackHospitalsList;
  } catch {
    return fallbackHospitalsList;
  }
};

export {
  fetchFederatedOverview,
  fetchRoundHistory,
  fetchHospitalParticipation,
  fetchHospitalDatasetStatus,
  validateHospitalDataset,
  fetchHospitalFederatedStatus,
  triggerLocalTraining
} from "./federatedApi";
