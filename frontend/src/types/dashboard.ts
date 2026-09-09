// frontend/src/types/dashboard.ts

export interface Hospital {
  id: number;
  hospital_code: string;
  name: string;
  region?: string;
  is_active: boolean;
  tier?: "enterprise" | "standard" | "research";
  api_key?: string;
  contact_email?: string;
  last_sync?: string;
  total_scans?: number;
}

export interface PatientRecord {
  id: string;
  name: string;
  phone: string;
  blood_group: string;
  admitted_date: string;
  last_inspected_date: string | null;
  ml_risk: {
    status: "completed" | "pending";
    level?: "Low" | "Moderate" | "High";
    score?: number;
  };
  dl_imaging: {
    status: "completed" | "pending";
    result?: "Stone" | "Normal";
    confidence?: number;
  };
  documents?: {
    name: string;
    size: string;
    uploaded_at: string;
  }[];
}

export interface PatientHistoryItem {
  id: number;
  reference_code: string;
  prediction_type: "risk" | "image";
  model_name: string;
  result_label: string | null;
  confidence: number | null;
  created_at: string;
}

export interface ModelPerformance {
  id: number;
  model_family: string;
  version_tag: string;
  accuracy: number | null;
  f1_score: number | null;
  mcc: number | null;
  is_deployed: boolean;
  trained_at: string;
  latency_ms?: number;
  environment?: "production" | "staging" | "canary";
  rollout_pct?: number;
}

export interface HospitalUpdateLogEntry {
  id?: number;
  hospital_name: string;
  version_tag: string;
  status: "received" | "pending" | "failed" | "quarantined";
  created_at: string;
  payload_size_mb?: number;
  gradient_hash?: string;
  latency_ms?: number;
}

export interface SystemLogEntry {
  hospital_name: string | null;
  level: "info" | "warning" | "error" | "critical";
  message: string;
  created_at: string;
  service?: string;
}

export interface DriftPoint {
  model_family: string;
  drift_score: number;
  metric_name: string;
  computed_at: string;
  status?: "normal" | "warning" | "drift_detected";
  p_value?: number;
  reference_mean?: number;
  current_mean?: number;
}

export interface SystemMonitoringSummary {
  total_predictions_24h: number;
  error_count_24h: number;
  avg_latency_ms: number | null;
  p95_latency_ms?: number;
  p99_latency_ms?: number;
  uptime_pct: number;
  gpu_utilization_pct?: number;
  cpu_utilization_pct?: number;
  active_nodes?: number;
}
