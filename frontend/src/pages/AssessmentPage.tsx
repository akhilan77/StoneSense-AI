import { useState } from 'react';
import { assess } from '../services/api';
import type { AssessmentResponse } from '../services/api';
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

const acceptedMimeTypes = ['image/jpeg', 'image/png', 'image/jpg'];

export function AssessmentPage() {
  const [patientData, setPatientData] = useState(defaultPatient);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [assessment, setAssessment] = useState<AssessmentResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0] ?? null;
    if (!file) {
      setImageFile(null);
      setPreviewUrl(null);
      return;
    }

    if (!acceptedMimeTypes.includes(file.type)) {
      setError('Please upload a JPG, JPEG, or PNG image.');
      return;
    }

    setError(null);
    setImageFile(file);
    setPreviewUrl(URL.createObjectURL(file));
  };

  const handleAssess = async () => {
    setIsLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('patient_data', JSON.stringify(patientData));
    if (imageFile) {
      formData.append('image', imageFile);
    }

    try {
      const response = await assess(formData);
      setAssessment(response);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : 'Unable to complete assessment.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <section className="rounded-3xl border border-slate-800 bg-slate-900/50 p-6 shadow-lg shadow-slate-950/20">
        <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
          <div>
            <h1 className="text-2xl font-extrabold text-white tracking-wide">Multi-Modal Diagnostic Assessment</h1>
            <p className="mt-2 text-sm text-slate-400">Aggregates deep learning CT classifiers and machine learning urine risk predictors into a unified diagnostic report.</p>
          </div>
          <button
            type="button"
            disabled={isLoading}
            onClick={handleAssess}
            className="rounded-full bg-cyan-400 hover:bg-cyan-300 active:scale-95 px-6 py-2.5 text-sm font-bold text-slate-950 transition-all shadow-md shadow-cyan-400/10 disabled:cursor-not-allowed disabled:bg-slate-700 disabled:text-slate-400"
          >
            {isLoading ? 'Assessing...' : 'Run assessment'}
          </button>
        </div>
      </section>

      {isLoading ? <LoadingSpinner label="Running Multi-Modal AI Analysis..." /> : null}
      {error ? <ErrorMessage message={error} /> : null}

      <section className="grid gap-6 md:grid-cols-2">
        {/* Left Column: Image Upload, Preview, and CNN results */}
        <div className="space-y-6">
          <div className="rounded-3xl border border-slate-800 bg-slate-900/40 p-6 shadow-xl shadow-slate-950/20">
            <h2 className="text-lg font-bold text-white mb-4 tracking-wide">CT Scan Input & Saliency Overlay</h2>
            
            <label className="flex cursor-pointer flex-col items-center justify-center rounded-3xl border border-dashed border-cyan-400/40 bg-slate-950/80 p-6 text-center hover:border-cyan-400/80 transition-colors mb-4">
              <span className="text-sm font-semibold text-cyan-300">Choose CT Scan Image</span>
              <span className="mt-1 text-xs text-slate-500">Supported: jpg, jpeg, png</span>
              <input
                type="file"
                accept=".jpg,.jpeg,.png,image/jpeg,image/png"
                className="hidden"
                onChange={handleFileChange}
              />
            </label>

            {previewUrl ? (
              <div className="relative overflow-hidden rounded-2xl border border-slate-800 bg-slate-950">
                <img src={previewUrl} alt="CT preview" className="max-h-80 w-full object-contain mx-auto" />
                <div className="absolute top-3 left-3 bg-slate-950/80 text-xs font-bold text-cyan-400 px-2.5 py-1 rounded-full border border-slate-850">
                  Input CT Slice
                </div>
              </div>
            ) : (
              <div className="rounded-2xl border border-dashed border-slate-800 p-8 text-center text-xs text-slate-500 bg-slate-950/30">
                No CT image uploaded. Load scan to enable deep learning spatial classification.
              </div>
            )}
          </div>

          {assessment?.ct_prediction ? (
            <PredictionCard
              title="Convolutional Neural Network (ResNet18)"
              data={[
                { label: 'Predicted Class', value: assessment.ct_prediction.class_name },
                { label: 'Classification Confidence', value: `${(assessment.ct_prediction.confidence * 100).toFixed(1)}%` },
              ]}
            />
          ) : null}
        </div>

        {/* Right Column: Tabular parameters, Risk prediction, and SHAP */}
        <div className="space-y-6">
          <div className="rounded-3xl border border-slate-800 bg-slate-900/40 p-6 shadow-xl shadow-slate-950/20">
            <h2 className="text-lg font-bold text-white mb-4 tracking-wide">Urine Biochemistry Parameters</h2>
            <div className="grid gap-4 grid-cols-2">
              <label className="text-xs font-semibold text-slate-450 uppercase tracking-wider">
                Calcium (calc)
                <input
                  type="number"
                  className="mt-2 w-full rounded-2xl border border-slate-700 bg-slate-950 px-4 py-2.5 text-slate-100 outline-none focus:border-cyan-400"
                  value={patientData.calcium}
                  onChange={(e) => setPatientData(prev => ({ ...prev, calcium: Number(e.target.value) }))}
                />
              </label>
              <label className="text-xs font-semibold text-slate-450 uppercase tracking-wider">
                Specific Gravity (gravity)
                <input
                  type="number"
                  step="0.01"
                  className="mt-2 w-full rounded-2xl border border-slate-700 bg-slate-950 px-4 py-2.5 text-slate-100 outline-none focus:border-cyan-400"
                  value={patientData.urine_specific_gravity}
                  onChange={(e) => setPatientData(prev => ({ ...prev, urine_specific_gravity: Number(e.target.value) }))}
                />
              </label>
            </div>
          </div>

          {assessment ? (
            <div className="space-y-6">
              <PredictionCard
                title="Tabular Risk Assessment (XGBoost)"
                data={[
                  { label: 'Stone Risk Probability', value: `${(assessment.risk_prediction.probability * 100).toFixed(1)}%` },
                  { label: 'Calculated Risk Level', value: assessment.risk_prediction.risk_level },
                ]}
              />

              {assessment.shap ? (
                <PredictionCard
                  title="SHAP Local Attributions"
                  data={[
                    { label: 'Calcium contribution', value: assessment.shap.feature_contributions.calc?.toFixed(4) ?? '0' },
                    { label: 'Gravity contribution', value: assessment.shap.feature_contributions.gravity?.toFixed(4) ?? '0' },
                    { label: 'pH contribution', value: assessment.shap.feature_contributions.ph?.toFixed(4) ?? '0' },
                  ]}
                />
              ) : null}

              <section className="rounded-3xl border border-slate-800 bg-gradient-to-br from-slate-900/90 to-slate-950/90 p-6 shadow-xl shadow-slate-950/40">
                <div className="flex items-center justify-between border-b border-slate-800 pb-3 mb-4">
                  <h3 className="text-md font-bold text-white tracking-wide">Clinical Recommendation</h3>
                  <span className="text-xs font-semibold text-slate-500">Latency: {assessment.processing_time_sec}s</span>
                </div>
                <p className="text-sm text-slate-350 leading-relaxed">{assessment.recommendation}</p>
              </section>
            </div>
          ) : (
            <div className="rounded-3xl border border-dashed border-slate-800 bg-slate-900/20 p-6 text-sm text-slate-500 text-center">
              Submit patient parameters to review risk predictions and local SHAP feature explanations.
            </div>
          )}
        </div>
      </section>
    </div>
  );
}
