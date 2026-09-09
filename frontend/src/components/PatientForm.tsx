import { useMemo, useState } from 'react';
import { predictRisk } from '../services/api';
import type { Patient } from '../types/patient';
import type { RiskPrediction } from '../types/riskPrediction';
import { ErrorMessage } from './ErrorMessage';
import { LoadingSpinner } from './LoadingSpinner';
import { PredictionCard } from './PredictionCard';
import { useHospital } from '../context/HospitalContext';

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
  osmolality: 550,
  conductivity: 22,
  urea: 250,
};

const fieldClassName =
  'mt-1 w-full rounded-md border border-stonesense-line bg-white px-3 py-1.5 text-sm text-stonesense-ink outline-none transition focus:border-stonesense-teal';

export function PatientForm() {
  const { hospitalId } = useHospital();
  const [patient, setPatient] = useState<Patient>(initialPatient);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<RiskPrediction | null>(null);

  const predictionData = useMemo(
    () => [
      { label: 'Probability', value: `${((prediction?.probability ?? 0) * 100).toFixed(0)}%` },
      { label: 'Risk Level', value: prediction?.risk_level ?? '—' },
      { label: 'Model', value: 'XGBoost' },
    ],
    [prediction]
  );

  const handleChange = (field: keyof Patient, value: number | string | boolean) => {
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
      const payload = {
        ...patient,
        hospital_id: hospitalId ?? 1,
      };
      const response = await predictRisk(payload);
      setPrediction(response);
    } catch (submitError) {
      setError(
        submitError instanceof Error ? submitError.message : 'Unable to fetch risk prediction.'
      );
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="grid gap-6 xl:grid-cols-[1.15fr_0.85fr]">
      <form
        onSubmit={handleSubmit}
        className="rounded-lg border border-stonesense-line bg-white p-6 shadow-sm"
      >
        <div className="mb-5 flex items-center justify-between gap-3">
          <div>
            <h2 className="font-serif text-lg text-stonesense-ink">Clinical Input</h2>
            <p className="text-sm text-stonesense-ink/60 mt-0.5">
              Complete the biomarker profile to request a risk assessment.
            </p>
          </div>
          <button
            type="submit"
            disabled={isLoading}
            className="rounded-md bg-stonesense-teal px-4 py-2 text-sm font-medium text-white transition hover:bg-stonesense-teal/90 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {isLoading ? 'Submitting...' : 'Run assessment'}
          </button>
        </div>

        <div className="grid gap-4 sm:grid-cols-2">
          <label className="text-xs text-stonesense-ink/60">
            Age
            <input
              type="number"
              className={fieldClassName}
              value={patient.age}
              onChange={(event) => handleChange('age', Number(event.target.value))}
            />
          </label>
          <label className="text-xs text-stonesense-ink/60">
            Gender
            <select
              className={fieldClassName}
              value={patient.gender}
              onChange={(event) => handleChange('gender', event.target.value)}
            >
              <option value="male">Male</option>
              <option value="female">Female</option>
              <option value="other">Other</option>
            </select>
          </label>
          <label className="text-xs text-stonesense-ink/60">
            BMI
            <input
              type="number"
              step="0.1"
              className={fieldClassName}
              value={patient.bmi}
              onChange={(event) => handleChange('bmi', Number(event.target.value))}
            />
          </label>
          <label className="text-xs text-stonesense-ink/60">
            Blood Pressure
            <input
              type="number"
              step="0.1"
              className={fieldClassName}
              value={patient.blood_pressure}
              onChange={(event) => handleChange('blood_pressure', Number(event.target.value))}
            />
          </label>
          <label className="text-xs text-stonesense-ink/60">
            Diabetes
            <select
              className={fieldClassName}
              value={String(patient.diabetes)}
              onChange={(event) => handleChange('diabetes', event.target.value === 'true')}
            >
              <option value="false">No</option>
              <option value="true">Yes</option>
            </select>
          </label>
          <label className="text-xs text-stonesense-ink/60">
            Family History
            <select
              className={fieldClassName}
              value={String(patient.family_history)}
              onChange={(event) => handleChange('family_history', event.target.value === 'true')}
            >
              <option value="false">No</option>
              <option value="true">Yes</option>
            </select>
          </label>
          <label className="text-xs text-stonesense-ink/60">
            Water Intake
            <input
              type="number"
              step="0.1"
              className={fieldClassName}
              value={patient.water_intake}
              onChange={(event) => handleChange('water_intake', Number(event.target.value))}
            />
          </label>
          <label className="text-xs text-stonesense-ink/60">
            Urine pH
            <input
              type="number"
              step="0.1"
              className={fieldClassName}
              value={patient.urine_ph}
              onChange={(event) => handleChange('urine_ph', Number(event.target.value))}
            />
          </label>
          <label className="text-xs text-stonesense-ink/60">
            Urine Specific Gravity
            <input
              type="number"
              step="0.01"
              className={fieldClassName}
              value={patient.urine_specific_gravity}
              onChange={(event) =>
                handleChange('urine_specific_gravity', Number(event.target.value))
              }
            />
          </label>
          <label className="text-xs text-stonesense-ink/60">
            Calcium
            <input
              type="number"
              step="0.1"
              className={fieldClassName}
              value={patient.calcium}
              onChange={(event) => handleChange('calcium', Number(event.target.value))}
            />
          </label>
          <label className="text-xs text-stonesense-ink/60">
            Uric Acid
            <input
              type="number"
              step="0.1"
              className={fieldClassName}
              value={patient.uric_acid}
              onChange={(event) => handleChange('uric_acid', Number(event.target.value))}
            />
          </label>
          <label className="text-xs text-stonesense-ink/60">
            Creatinine
            <input
              type="number"
              step="0.1"
              className={fieldClassName}
              value={patient.creatinine}
              onChange={(event) => handleChange('creatinine', Number(event.target.value))}
            />
          </label>
          <label className="text-xs text-stonesense-ink/60">
            Osmolality
            <input
              type="number"
              step="0.1"
              className={fieldClassName}
              value={patient.osmolality}
              onChange={(event) => handleChange('osmolality', Number(event.target.value))}
            />
          </label>
          <label className="text-xs text-stonesense-ink/60">
            Conductivity
            <input
              type="number"
              step="0.1"
              className={fieldClassName}
              value={patient.conductivity}
              onChange={(event) => handleChange('conductivity', Number(event.target.value))}
            />
          </label>
          <label className="text-xs text-stonesense-ink/60">
            Urea
            <input
              type="number"
              step="0.1"
              className={fieldClassName}
              value={patient.urea}
              onChange={(event) => handleChange('urea', Number(event.target.value))}
            />
          </label>
        </div>

        {isLoading ? (
          <div className="mt-4">
            <LoadingSpinner label="Running risk prediction..." />
          </div>
        ) : null}
        {error ? (
          <div className="mt-4">
            <ErrorMessage message={error} />
          </div>
        ) : null}
      </form>

      <div className="space-y-4">
        {prediction ? (
          <div className="space-y-4">
            <PredictionCard title="Clinical Risk Assessment" data={predictionData} />
            <section className="rounded-lg border border-stonesense-line bg-white p-6 shadow-sm">
              <p className="text-xs font-semibold uppercase tracking-wider text-stonesense-teal">
                Clinical Evidence
              </p>
              <h2 className="mt-1 font-serif text-lg text-stonesense-ink">Trustworthy Assessment</h2>
              <h3 className="mt-3 font-serif text-base text-stonesense-ink">Key contributing factors</h3>
              <ol className="mt-2 list-decimal space-y-1.5 pl-5 text-sm text-stonesense-ink/80">
                {prediction.shap?.top_features.slice(0, 4).map((feature) => (
                  <li key={feature}>
                    {feature}
                    <span className="ml-2 text-stonesense-ink/50">
                      {prediction.shap?.feature_directions[feature]} the model output
                    </span>
                  </li>
                ))}
              </ol>
              <p className="mt-4 text-sm text-stonesense-ink/60">
                {prediction.shap?.summary ?? 'SHAP explanations are unavailable for this response.'}
              </p>
              <div className="mt-5 grid gap-3 border-t border-stonesense-line pt-4 sm:grid-cols-2">
                <div>
                  <p className="text-xs text-stonesense-ink/50">Model</p>
                  <p className="font-semibold text-stonesense-ink">XGBoost</p>
                </div>
                <div>
                  <p className="text-xs text-stonesense-ink/50">Explainability</p>
                  <p className="font-semibold text-stonesense-ink">SHAP</p>
                </div>
                <div>
                  <p className="text-xs text-stonesense-ink/50">Input</p>
                  <p className="font-semibold text-stonesense-ink">Clinical / Tabular Data</p>
                </div>
                <div>
                  <p className="text-xs text-stonesense-ink/50">Assessment type</p>
                  <p className="font-semibold text-stonesense-ink">Clinical Risk Assessment</p>
                </div>
              </div>
            </section>
          </div>
        ) : (
          <div className="rounded-lg border border-dashed border-stonesense-line bg-white p-6 text-center text-sm text-stonesense-ink/50">
            Submit a patient profile to receive a risk prediction response.
          </div>
        )}
      </div>
    </div>
  );
}
