export type ConcordanceOutcomeCode =
  | 'CONCORDANT_STONE_EVIDENCE'
  | 'CONCORDANT_NO_STONE_EVIDENCE'
  | 'CONCORDANT_LOW_CONCERN' // Legacy code compatibility
  | 'DISCORDANT'
  | 'NON_STONE_FINDING'
  | 'LOW_CONFIDENCE'
  | 'INSUFFICIENT_DATA';

export interface TabularRiskInput {
  probability?: number;
  risk_level?: string;
  confidence?: number;
}

export interface ImagingOutputInput {
  class_name?: string;
  confidence?: number;
  probabilities?: Record<string, number>;
  status?: string;
}

export interface ConcordanceRequest {
  risk_prediction_id?: number | null;
  image_prediction_id?: number | null;
  patient_id?: number | null;
  risk?: TabularRiskInput | null;
  imaging?: ImagingOutputInput | null;
}

export interface ConcordanceResult {
  outcome_code: ConcordanceOutcomeCode;
  outcome_label: string;
  explanation: string;
  rule_fired: string;
  clinician_review_recommended: boolean;
  inputs_used: {
    risk?: Record<string, unknown> | null;
    imaging?: Record<string, unknown> | null;
  };
  thresholds_applied: {
    risk_threshold?: number;
    tabular_min_confidence?: number;
    imaging_min_confidence?: number;
  };
  disclaimer: string;
}
