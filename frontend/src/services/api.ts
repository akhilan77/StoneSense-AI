import axios from 'axios';
import type { AssessmentResponse } from '../types/assessment';
import type { Patient } from '../types/patient';
import type { RiskPrediction } from '../types/riskPrediction';
import type { StoneDetection } from '../types/stoneDetection';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export async function health(): Promise<{ status: string }> {
  const response = await api.get<{ status: string }>('/health');
  return response.data;
}

export async function predictRisk(payload: Patient): Promise<RiskPrediction> {
  const response = await api.post<RiskPrediction>('/predict/risk', payload);
  return response.data;
}

export async function predictImage(formData: FormData): Promise<StoneDetection> {
  const response = await api.post<StoneDetection>('/predict/image', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
}

export async function assess(payload: Patient): Promise<AssessmentResponse> {
  const response = await api.post<AssessmentResponse>('/assess', payload);
  return response.data;
}

