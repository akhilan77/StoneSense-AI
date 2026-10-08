export type DLModelStatus = 'READY' | 'PENDING_WEIGHTS' | 'ERROR';

export interface DLModelMetadata {
  id: string;
  name: string;
  family: string;
  status: DLModelStatus;
  is_ready: boolean;
  accuracy?: number | null;
  macro_f1?: number | null;
  stone_recall?: number | null;
  tumor_recall?: number | null;
  classes?: string[];
  version_tag?: string;
  display_name?: string;
  error_message?: string | null;
  is_deployed?: boolean;
}

export interface DLModelListResponse {
  models: DLModelMetadata[];
}

export interface GradCAMResult {
  overlay_url: string;
  target_class?: string;
  available?: boolean;
  message?: string;
}

export interface StoneDetection {
  class_name: string;
  confidence: number;
  inference_time_sec: number;
  model_id?: string;
  model_name?: string;
  gradcam?: GradCAMResult;
}
