import { apiClient } from './apiClient';
import type { ConcordanceRequest, ConcordanceResult } from '../types/concordance';

export async function predictConcordance(payload: ConcordanceRequest): Promise<ConcordanceResult> {
  const response = await apiClient.post<ConcordanceResult>('/api/v1/predict/concordance', payload);
  return response.data;
}
