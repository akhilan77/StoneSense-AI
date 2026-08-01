import { useState } from 'react';
import { assess } from '../services/api';
import type { AssessmentResponse } from '../types/assessment';
import { LoadingSpinner } from '../components/LoadingSpinner';
import { ErrorMessage } from '../components/ErrorMessage';
import { PredictionCard } from '../components/PredictionCard';

const defaultPatient = {
  age: 42,
  gender: 'male' as const,
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

export function AssessmentPage() {
  const [assessment, setAssessment] = useState<AssessmentResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleAssess = async () => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await assess(defaultPatient);
      setAssessment(response);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : 'Unable to complete assessment.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <section className="rounded-3xl border border-slate-800 bg-slate-900/50 p-5">
        <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
          <div>
            <h1 className="text-2xl font-semibold text-white">Assessment</h1>
            <p className="mt-2 text-sm text-slate-400">Request the final composite assessment payload containing patient, risk, stone detection, explainability, and recommendation.</p>
          </div>
          <button
            type="button"
            disabled={isLoading}
            onClick={handleAssess}
            className="rounded-full bg-cyan-400 px-5 py-2 text-sm font-semibold text-slate-950 transition hover:bg-cyan-300 disabled:cursor-not-allowed disabled:bg-slate-700 disabled:text-slate-400"
          >
            {isLoading ? 'Assessing...' : 'Run assessment'}
          </button>
        </div>
      </section>

      {isLoading ? <LoadingSpinner label="Building assessment payload..." /> : null}
      {error ? <ErrorMessage message={error} /> : null}

      {assessment ? (
        <div className="grid gap-5 lg:grid-cols-2">
          <PredictionCard
            title="Patient"
            data={[
              { label: 'Age', value: assessment.patient.age },
              { label: 'Gender', value: assessment.patient.gender },
              { label: 'BMI', value: assessment.patient.bmi },
              { label: 'Blood Pressure', value: assessment.patient.blood_pressure },
              { label: 'Diabetes', value: assessment.patient.diabetes },
              { label: 'Family History', value: assessment.patient.family_history },
            ]}
          />
          <PredictionCard
            title="Risk Prediction"
            data={[
              { label: 'Probability', value: `${(assessment.risk_prediction.probability * 100).toFixed(0)}%` },
              { label: 'Risk Level', value: assessment.risk_prediction.risk_level },
              { label: 'Confidence', value: `${(assessment.risk_prediction.confidence * 100).toFixed(0)}%` },
            ]}
          />
          <PredictionCard
            title="Stone Detection"
            data={[
              { label: 'Detected', value: assessment.stone_detection.detected },
              { label: 'Confidence', value: `${(assessment.stone_detection.confidence * 100).toFixed(0)}%` },
              { label: 'Stone Size', value: assessment.stone_detection.stone_size },
              { label: 'Stone Location', value: assessment.stone_detection.stone_location },
            ]}
          />
          <PredictionCard
            title="Explainability"
            data={[
              { label: 'LIME', value: assessment.explainability.lime_explanation },
              { label: 'GradCAM URL', value: assessment.explainability.gradcam_image_url },
              { label: 'SHAP Age', value: assessment.explainability.shap_values.age ?? 0 },
              { label: 'SHAP BMI', value: assessment.explainability.shap_values.bmi ?? 0 },
            ]}
          />
          <section className="rounded-3xl border border-slate-800 bg-slate-900/80 p-5 shadow-xl shadow-slate-950/40 lg:col-span-2">
            <h3 className="text-lg font-semibold text-white">Recommendation</h3>
            <p className="mt-2 text-slate-300">{assessment.recommendation}</p>
          </section>
        </div>
      ) : (
        <div className="rounded-3xl border border-dashed border-slate-700 bg-slate-900/40 p-6 text-sm text-slate-400">
          Use the assessment action to invoke the mock API contract and render the final response as cards.
        </div>
      )}
    </div>
  );
}
