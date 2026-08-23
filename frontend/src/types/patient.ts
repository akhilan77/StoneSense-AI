export interface Patient {
  age: number;
  gender: 'male' | 'female' | 'other';
  bmi: number;
  blood_pressure: number;
  diabetes: boolean;
  family_history: boolean;
  water_intake: number;
  urine_ph: number;
  urine_specific_gravity: number;
  calcium: number;
  uric_acid: number;
  creatinine: number;
  osmolality: number;
  conductivity: number;
  urea: number;
}
