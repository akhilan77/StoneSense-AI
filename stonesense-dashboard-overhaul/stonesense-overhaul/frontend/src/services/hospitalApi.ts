// Drop at: frontend/src/services/hospitalApi.ts
import axios from "axios";
import { Hospital, PatientHistoryItem } from "../types/dashboard";

const client = axios.create({ baseURL: "/api/v1/hospital" });

export async function fetchHospitals(): Promise<Hospital[]> {
  const { data } = await client.get<Hospital[]>("/list");
  return data;
}

export async function fetchHistory(hospitalId: number, limit = 25): Promise<PatientHistoryItem[]> {
  const { data } = await client.get<PatientHistoryItem[]>(`/${hospitalId}/history`, {
    params: { limit },
  });
  return data;
}
