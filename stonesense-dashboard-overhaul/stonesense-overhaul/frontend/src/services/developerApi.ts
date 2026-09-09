// Drop at: frontend/src/services/developerApi.ts
import axios from "axios";
import {
  ModelPerformance,
  HospitalUpdateLogEntry,
  SystemLogEntry,
  DriftPoint,
  SystemMonitoringSummary,
} from "../types/dashboard";

const client = axios.create({ baseURL: "/api/v1/developer" });

export const fetchModelPerformance = async (): Promise<ModelPerformance[]> =>
  (await client.get<ModelPerformance[]>("/model-performance")).data;

export const fetchModelVersions = async (modelFamily?: string): Promise<ModelPerformance[]> =>
  (await client.get<ModelPerformance[]>("/model-versions", { params: { model_family: modelFamily } })).data;

export const deployModelVersion = async (modelVersionId: number) =>
  (await client.post("/model-versions/deploy", { model_version_id: modelVersionId })).data;

export const fetchHospitalLogs = async (): Promise<HospitalUpdateLogEntry[]> =>
  (await client.get<HospitalUpdateLogEntry[]>("/hospital-logs")).data;

export const fetchSystemMonitoring = async (): Promise<SystemMonitoringSummary> =>
  (await client.get<SystemMonitoringSummary>("/system-monitoring")).data;

export const fetchSystemLogs = async (limit = 30): Promise<SystemLogEntry[]> =>
  (await client.get<SystemLogEntry[]>("/system-logs", { params: { limit } })).data;

export const fetchDriftAnalysis = async (modelFamily?: string): Promise<DriftPoint[]> =>
  (await client.get<DriftPoint[]>("/drift-analysis", { params: { model_family: modelFamily } })).data;
