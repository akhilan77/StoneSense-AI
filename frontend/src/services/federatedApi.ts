// frontend/src/services/federatedApi.ts
import axios from "axios";
import {
  FederatedOverview,
  FederatedRoundDetail,
  HospitalParticipation,
  DatasetStatus,
  DatasetValidationResult,
  FederatedStatus,
  RoundLiveStatus,
  HospitalLiveStatus,
  StartRoundResponse,
  FederatedEventMessage,
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

export async function fetchHospitalTrainingHistory(hospitalId: number): Promise<FederatedRoundDetail["hospital_runs"]> {
  const { data } = await client.get<FederatedRoundDetail["hospital_runs"]>(`/hospital/${hospitalId}/training-history`);
  return data;
}

export async function fetchHospitalCurrentModel(hospitalId: number): Promise<{
  hospital_id: number;
  hospital_code: string;
  current_model_version: string;
  global_round: number;
  global_accuracy: number | null;
  global_f1: number | null;
  deployed_at: string;
}> {
  const { data } = await client.get(`/hospital/${hospitalId}/current-model`);
  return data;
}

export async function triggerLocalTraining(hospitalId: number): Promise<{ message: string; status: string }> {
  const { data } = await client.post(`/hospital/${hospitalId}/train-local`);
  return data;
}

export async function startFederatedRound(params?: {
  num_rounds?: number;
  local_epochs?: number;
  batch_size?: number;
  lr?: number;
  mode?: string;
}): Promise<StartRoundResponse> {
  const { data } = await client.post<StartRoundResponse>("/developer/federated/rounds/start", params || {});
  return data;
}

export async function fetchCurrentRoundLiveStatus(): Promise<RoundLiveStatus> {
  const { data } = await client.get<RoundLiveStatus>("/developer/federated/rounds/current/status");
  return data;
}

export async function fetchRoundStatusById(roundId: number): Promise<RoundLiveStatus> {
  const { data } = await client.get<RoundLiveStatus>(`/developer/federated/rounds/${roundId}/status`);
  return data;
}

export async function fetchHospitalLiveStatus(hospitalId: number | string): Promise<HospitalLiveStatus> {
  const { data } = await client.get<HospitalLiveStatus>(`/hospital/${hospitalId}/federated-live-status`);
  return data;
}

export function createFederatedWebSocket(
  onMessage: (msg: FederatedEventMessage) => void,
  onError?: (err: Event) => void,
  hospitalCode?: string,
  onConnectionChange?: (connected: boolean) => void
): () => void {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const host = window.location.host;
  const wsUrl = hospitalCode
    ? `${protocol}//${host}/api/v1/hospital/${encodeURIComponent(hospitalCode)}/federated/ws`
    : `${protocol}//${host}/api/v1/developer/federated/ws`;

  let ws: WebSocket | null = null;
  let keepAliveInterval: any = null;

  try {
    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      onConnectionChange?.(true);
      keepAliveInterval = setInterval(() => {
        if (ws && ws.readyState === WebSocket.OPEN) {
          ws.send("ping");
        }
      }, 15000);
    };

    ws.onmessage = (event) => {
      try {
        const payload: FederatedEventMessage = JSON.parse(event.data);
        if (payload.event !== "pong") {
          onMessage(payload);
        }
      } catch (err) {
        console.warn("Error parsing WebSocket event:", err);
      }
    };

    ws.onerror = (err) => {
      if (onError) onError(err);
    };
    ws.onclose = () => onConnectionChange?.(false);
  } catch (err) {
    onConnectionChange?.(false);
    console.warn("WebSocket connection could not be established:", err);
  }

  return () => {
    if (keepAliveInterval) clearInterval(keepAliveInterval);
    if (ws) {
      ws.close();
      ws = null;
    }
  };
}

