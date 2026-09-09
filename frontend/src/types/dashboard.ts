// Drop at: frontend/src/types/dashboard.ts

export interface Hospital {
  id: number;
  hospital_code: string;
  name: string;
  region?: string;
  is_active: boolean;
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
}

export interface HospitalUpdateLogEntry {
  hospital_name: string;
  version_tag: string;
  status: "received" | "pending" | "failed";
  created_at: string;
}

export interface SystemLogEntry {
  hospital_name: string | null;
  level: "info" | "warning" | "error";
  message: string;
  created_at: string;
}

export interface DriftPoint {
  model_family: string;
  drift_score: number;
  metric_name: string;
  computed_at: string;
}

export interface SystemMonitoringSummary {
  total_predictions_24h: number;
  error_count_24h: number;
  avg_latency_ms: number | null;
  uptime_pct: number;
}
