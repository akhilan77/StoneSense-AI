import { useMemo, useState } from 'react';
import type { Patient } from '../types/patient';
import type { RiskPrediction } from '../types/riskPrediction';
import { LoadingSpinner } from './LoadingSpinner';
import { ErrorMessage } from './ErrorMessage';
import { PredictionCard } from './PredictionCard';
import { predictRisk } from '../services/api';

const initialPatient: Patient = {
  age: 42,
  gender: 'male',
  bmi: 27.4,
  blood_pressure: 132,
  diabetes: false,
  family_history: false,
  water_intake: 2.1,
  urine_ph: 6.1,
  urine_specific_gravity: 1.02,
  calcium: 24,
  uric_acid: 6.5,
  creatinine: 0.9,
};

const fieldClassName =
  'mt-2 w-full rounded-2xl border border-slate-700 bg-slate-950 px-3 py-2 text-slate-100 outline-none transition focus:border-cyan-400';

export function PatientForm() {
  const [patient, setPatient] = useState<Patient>(initialPatient);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<RiskPrediction | null>(null);

  const predictionData = useMemo(
    () => [
      { label: 'Probability', value: `${((prediction?.probability ?? 0) * 100).toFixed(0)}%` },
      { label: 'Risk Level', value: prediction?.risk_level ?? '—' },
      { label: 'Confidence', value: `${((prediction?.confidence ?? 0) * 100).toFixed(0)}%` },
    ],
    [prediction],
  );

  const handleChange = (field: keyof Patient, value: string | boolean) => {
    setPatient((current) => ({
      ...current,
      [field]: field === 'gender' ? (value as Patient['gender']) : value,
    }));
  };

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setIsLoading(true);
    setError(null);

    try {
      const response = await predictRisk(patient);
      setPrediction(response);
    } catch (submitError) {
      setError(submitError instanceof Error ? submitError.message : 'Unable to fetch risk prediction.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="grid gap-6 xl:grid-cols-[1.15fr_0.85fr]">
      <form onSubmit={handleSubmit} className="rounded-3xl border border-slate-800 bg-slate-900/70 p-5 shadow-xl shadow-slate-950/40">
        <div className="mb-5 flex items-center justify-between gap-3">
          <div>
            <h2 className="text-xl font-semibold text-white">Patient Information</h2>
            <p className="text-sm text-slate-400">Complete the biomarker profile to request a mock risk assessment.</p>
          </div>
          <button
            type="submit"
            disabled={isLoading}
            className="rounded-full bg-cyan-400 px-4 py-2 text-sm font-semibold text-slate-950 transition hover:bg-cyan-300 disabled:cursor-not-allowed disabled:bg-slate-700 disabled:text-slate-400"
          >
            {isLoading ? 'Submitting...' : 'Submit'}
          </button>
        </div>

        <div className="grid gap-4 sm:grid-cols-2">
          <label className="text-sm text-slate-300">
            Age
            <input type="number" className={fieldClassName} value={patient.age} onChange={(event) => handleChange('age', Number(event.target.value))} />
          </label>
          <label className="text-sm text-slate-300">
            Gender
            <select className={fieldClassName} value={patient.gender} onChange={(event) => handleChange('gender', event.target.value)}>
              <option value="male">Male</option>
              <option value="female">Female</option>
              <option value="other">Other</option>
            </select>
          </label>
          <label className="text-sm text-slate-300">
            BMI
            <input type="number" step="0.1" className={fieldClassName} value={patient.bmi} onChange={(event) => handleChange('bmi', Number(event.target.value))} />
          </label>
          <label className="text-sm text-slate-300">
            Blood Pressure
            <input type="number" step="0.1" className={fieldClassName} value={patient.blood_pressure} onChange={(event) => handleChange('blood_pressure', Number(event.target.value))} />
          </label>
          <label className="text-sm text-slate-300">
            Diabetes
            <select className={fieldClassName} value={String(patient.diabetes)} onChange={(event) => handleChange('diabetes', event.target.value === 'true')}>
              <option value="false">No</option>
              <option value="true">Yes</option>
            </select>
          </label>
          <label className="text-sm text-slate-300">
            Family History
            <select className={fieldClassName} value={String(patient.family_history)} onChange={(event) => handleChange('family_history', event.target.value === 'true')}>
              <option value="false">No</option>
              <option value="true">Yes</option>
            </select>
          </label>
          <label className="text-sm text-slate-300">
            Water Intake
            <input type="number" step="0.1" className={fieldClassName} value={patient.water_intake} onChange={(event) => handleChange('water_intake', Number(event.target.value))} />
          </label>
          <label className="text-sm text-slate-300">
            Urine pH
            <input type="number" step="0.1" className={fieldClassName} value={patient.urine_ph} onChange={(event) => handleChange('urine_ph', Number(event.target.value))} />
          </label>
          <label className="text-sm text-slate-300">
            Urine Specific Gravity
            <input type="number" step="0.01" className={fieldClassName} value={patient.urine_specific_gravity} onChange={(event) => handleChange('urine_specific_gravity', Number(event.target.value))} />
          </label>
          <label className="text-sm text-slate-300">
            Calcium
            <input type="number" step="0.1" className={fieldClassName} value={patient.calcium} onChange={(event) => handleChange('calcium', Number(event.target.value))} />
          </label>
          <label className="text-sm text-slate-300">
            Uric Acid
            <input type="number" step="0.1" className={fieldClassName} value={patient.uric_acid} onChange={(event) => handleChange('uric_acid', Number(event.target.value))} />
          </label>
          <label className="text-sm text-slate-300">
            Creatinine
            <input type="number" step="0.1" className={fieldClassName} value={patient.creatinine} onChange={(event) => handleChange('creatinine', Number(event.target.value))} />
          </label>
        </div>

        {isLoading ? <div className="mt-4"><LoadingSpinner label="Running risk prediction..." /></div> : null}
        {error ? <div className="mt-4"><ErrorMessage message={error} /></div> : null}
      </form>

      <div className="space-y-4">
        {prediction ? <PredictionCard title="Prediction Output" data={predictionData} /> : <div className="rounded-3xl border border-dashed border-slate-700 bg-slate-900/40 p-6 text-sm text-slate-400">Submit a patient profile to receive a mock risk prediction response.</div>}
      </div>
    </div>
  );
}
