import type { AssessmentResponse } from '../types/assessment';
import type { Patient } from '../types/patient';
import type { RiskPrediction } from '../types/riskPrediction';
import type { StoneDetection } from '../types/stoneDetection';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000';

class StoneSenseApiService {
  private readonly baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  private async request<T>(path: string, options?: RequestInit): Promise<T> {
    const response = await fetch(`${this.baseUrl}${path}`, {
      headers: {
        'Content-Type': 'application/json',
        ...(options?.headers ?? {}),
      },
      ...options,
    });

    if (!response.ok) {
      throw new Error(`Request failed with status ${response.status}`);
    }

    return (await response.json()) as T;
  }

  async getRoot(): Promise<{ message: string }> {
    return this.request<{ message: string }>(`/`);
  }

  async getHealth(): Promise<{ status: string }> {
    return this.request<{ status: string }>(`/health`);
  }

  async predictRisk(payload: Patient): Promise<RiskPrediction> {
    return this.request<RiskPrediction>(`/predict/risk`, {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  }

  async predictImage(formData: FormData): Promise<StoneDetection> {
    return fetch(`${this.baseUrl}/predict/image`, {
      method: 'POST',
      body: formData,
    }).then(async (response) => {
      if (!response.ok) {
        throw new Error(`Request failed with status ${response.status}`);
      }

      return (await response.json()) as StoneDetection;
    });
  }

  async assess(payload: Patient): Promise<AssessmentResponse> {
    return this.request<AssessmentResponse>(`/assess`, {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  }
}

export const stoneSenseApiService = new StoneSenseApiService();
export default stoneSenseApiService;
