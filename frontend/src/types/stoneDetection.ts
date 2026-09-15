export interface StoneDetection {
  class_name: string;
  confidence: number;
  inference_time_sec: number;
  gradcam?: {
    overlay_url: string;
    target_class?: string;
    available?: boolean;
    message?: string;
  };
}
