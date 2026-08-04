import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface HealthResponse {
  status: string;
  version: string;
  models_loaded: {
    resnet18_ct_classification: boolean;
    xgboost_risk_prediction: boolean;
    preprocessing_pipeline: boolean;
  };
  hardware: {
    cuda_available: boolean;
    active_device: string;
  };
}

export interface ModelInfoResponse {
  dl_resnet18: {
    name: string;
    classes: string[];
    accuracy: number;
    f1_macro: number;
    target_layer: string;
    loaded: boolean;
  };
  ml_xgboost: {
    name: string;
    features: string[];
    validation_accuracy: number;
    validation_f1: number;
    validation_mcc: number;
    loaded: boolean;
  };
}

export interface RiskPrediction {
  probability: number;
  risk_level: 'Low' | 'High';
  confidence: number;
  inference_time_sec: number;
}

export interface StoneDetection {
  class_name: string;
  confidence: number;
  inference_time_sec: number;
}

export interface ShapExplanation {
  top_features: string[];
  feature_contributions: Record<string, number>;
}

export interface AssessmentResponse {
  risk_prediction: {
    probability: number;
    risk_level: 'Low' | 'High';
    confidence: number;
    inference_time_sec: number;
  };
  ct_prediction?: {
    class_name: string;
    confidence: number;
    inference_time_sec: number;
  };
  gradcam?: {
    overlay_path: string;
  };
  shap?: ShapExplanation;
  recommendation: string;
  processing_time_sec: number;
}

export async function health(): Promise<HealthResponse> {
  const response = await api.get<HealthResponse>('/api/v1/health');
  return response.data;
}

export async function predictRisk(payload: any): Promise<RiskPrediction> {
  const response = await api.post<RiskPrediction>('/api/v1/predict/risk', payload);
  return response.data;
}

export async function predictImage(formData: FormData): Promise<StoneDetection> {
  const response = await api.post<StoneDetection>('/api/v1/predict/image', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
}

export async function assess(formData: FormData): Promise<AssessmentResponse> {
  const response = await api.post<AssessmentResponse>('/api/v1/assessment', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
}

export async function getModelInfo(): Promise<ModelInfoResponse> {
  const response = await api.get<ModelInfoResponse>('/api/v1/models');
  return response.data;
}
