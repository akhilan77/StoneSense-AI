// frontend/src/types/federated.ts

export interface FederatedOverview {
  current_round: number;
  global_accuracy: number;
  global_f1: number;
  global_loss: number;
  global_precision: number;
  global_recall: number;
  active_hospitals_count: number;
  current_model_version: string;
  total_samples: number;
  last_updated: string;
}

export interface HospitalRunTelemetry {
  id: number;
  round_number: number;
  hospital_id: number;
  hospital_code: string;
  hospital_name?: string;
  model_version?: string;
  train_loss?: number;
  train_acc?: number;
  train_f1?: number;
  val_loss?: number;
  val_acc?: number;
  val_f1?: number;
  sample_count: number;
  duration_sec?: number;
  created_at: string;
}

export type HospitalFLStatus =
  | 'WAITING'
  | 'MODEL_RECEIVED'
  | 'TRAINING'
  | 'TRAINING_COMPLETED'
  | 'UPDATE_SUBMITTED'
  | 'WAITING_FOR_AGGREGATION'
  | 'MODEL_UPDATED'
  | 'COMPLETED'
  | 'FAILED';

export interface FederatedRoundDetail {
  id: number;
  round_number: number;
  mode: string;
  participants_count: number;
  global_train_loss?: number;
  global_train_acc?: number;
  global_val_loss?: number;
  global_val_acc?: number;
  global_val_f1?: number;
  global_val_precision?: number;
  global_val_recall?: number;
  duration_sec?: number;
  status: string;
  completed_at: string;
  hospital_runs: HospitalRunTelemetry[];
}

export interface HospitalParticipation {
  hospital_id: number;
  hospital_code: string;
  name: string;
  region?: string;
  dataset_size: number;
  total_rounds_participated: number;
  sample_contribution_pct: number;
  latest_local_f1?: number;
  current_model_version?: string;
}

export interface DatasetStatus {
  hospital_id: number;
  hospital_code: string;
  name: string;
  dataset_size: number;
  class_distribution: {
    Cyst: number;
    Normal: number;
    Stone: number;
    Tumor: number;
    [key: string]: number;
  };
  is_valid: boolean;
  split_info: {
    train?: number;
    validation?: number;
    test?: number;
    [key: string]: number | undefined;
  };
  dataset_version?: string;
  last_updated?: string;
  last_validated?: string;
  validation_message?: string;
}

export interface DatasetValidationResult {
  hospital_code: string;
  is_valid: boolean;
  total_samples: number;
  classes: Record<string, number>;
  corrupted_images: number;
  message: string;
}

export interface FederatedStatus {
  hospital_code: string;
  status: string;
  current_round: number;
  current_model_version: string;
  last_round_participated?: number;
  local_accuracy?: number;
  local_f1?: number;
}

export interface ClientLiveStatus {
  hospital_id: string;
  hospital_name?: string;
  status: HospitalFLStatus | string;
  samples?: number;
  accuracy?: number;
  f1?: number;
  loss?: number;
  duration_sec?: number;
}

export interface RoundLiveStatus {
  round: number;
  status:
    | 'READY'
    | 'ROUND_STARTED'
    | 'GLOBAL_MODEL_DISTRIBUTING'
    | 'LOCAL_TRAINING'
    | 'FEDAVG_STARTED'
    | 'FEDAVG_COMPLETED'
    | 'GLOBAL_MODEL_SAVED'
    | 'COMPLETED'
    | 'FAILED'
    | string;
  previous_model_version?: string;
  global_model_version: string;
  participating_hospitals: number;
  completed_hospitals: number;
  clients: ClientLiveStatus[];
  current_step?: string;
  error_message?: string;
  metrics?: {
    accuracy?: number;
    f1?: number;
    loss?: number;
    precision?: number;
    recall?: number;
  };
}

export interface HospitalLiveStatus {
  hospital_id: string;
  round: number;
  status: HospitalFLStatus;
  phase?: RoundLiveStatus['status'];
  global_model_version: string;
  update_submitted: boolean;
  model_updated: boolean;
  last_event?: string;
  last_event_at?: string;
  round_status?: RoundLiveStatus['status'];
  local_training?: {
    status: HospitalFLStatus;
    samples: number | null;
    accuracy?: number;
    f1?: number;
    loss?: number;
    duration_sec?: number;
  };
}

export interface FederatedEventMessage {
  event: string;
  round: number;
  hospital_id?: string;
  model_version?: string;
  previous_model_version?: string;
  status?: string;
  current_step?: string;
  data?: Record<string, any>;
  payload?: Record<string, any>;
  timestamp?: string;
}

export interface StartRoundResponse {
  round: number;
  status: string;
  global_model_version: string;
  message?: string;
}
