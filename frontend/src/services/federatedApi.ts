// frontend/src/services/federatedApi.ts
import axios from "axios";
import {
  FederatedOverview,
  FederatedRoundDetail,
  HospitalParticipation,
  DatasetStatus,
  DatasetValidationResult,
  FederatedStatus
} from "../types/federated";

const client = axios.create({ baseURL: "/api/v1" });

export async function fetchFederatedOverview(): Promise<FederatedOverview> {
  const { data } = await client.get<FederatedOverview>("/developer/federated-overview");
  return data;
}

export async function fetchRoundHistory(): Promise<FederatedRoundDetail[]> {
  const { data } = await client.get<FederatedRoundDetail[]>("/developer/round-history");
  return data;
}

export async function fetchHospitalParticipation(): Promise<HospitalParticipation[]> {
  const { data } = await client.get<HospitalParticipation[]>("/developer/hospital-participation");
  return data;
}

export async function fetchHospitalDatasetStatus(hospitalId: number): Promise<DatasetStatus> {
  const { data } = await client.get<DatasetStatus>(`/hospital/${hospitalId}/dataset-status`);
  return data;
}

export async function validateHospitalDataset(hospitalId: number): Promise<DatasetValidationResult> {
  const { data } = await client.post<DatasetValidationResult>(`/hospital/${hospitalId}/dataset-validate`);
  return data;
}

export async function fetchHospitalFederatedStatus(hospitalId: number): Promise<FederatedStatus> {
  const { data } = await client.get<FederatedStatus>(`/hospital/${hospitalId}/federated-status`);
  return data;
}

export async function triggerLocalTraining(hospitalId: number): Promise<{ message: string; status: string }> {
  const { data } = await client.post(`/hospital/${hospitalId}/train-local`);
  return data;
}
