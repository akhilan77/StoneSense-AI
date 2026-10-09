import { Hospital, PatientHistoryItem } from "../types/dashboard";
import { apiClient } from "./apiClient";

export async function fetchHospitals(): Promise<Hospital[]> {
  const { data } = await apiClient.get<Hospital[]>("/api/v1/hospital/list");
  return data;
}

export async function fetchHistory(hospitalId: number, limit = 25): Promise<PatientHistoryItem[]> {
  const { data } = await apiClient.get<PatientHistoryItem[]>(`/api/v1/hospital/${hospitalId}/history`, {
    params: { limit },
  });
  return data;
}
