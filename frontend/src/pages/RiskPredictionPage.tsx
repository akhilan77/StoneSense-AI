import { useState } from 'react';
import AppLayout from '../components/layout/AppLayout';
import { NhanesClinicalProfile } from '../components/NhanesClinicalProfile';
import { PatientForm } from '../components/PatientForm';
import { useHospital } from '../context/useHospital';
import { useParams } from 'react-router-dom';

type RiskMode = 'urine' | 'nhanes';

const modes: Array<{ key: RiskMode; label: string; hint: string }> = [
  { key: 'urine', label: 'Urine biomarkers', hint: 'Existing 6-feature urine model' },
  { key: 'nhanes', label: 'Clinical profile (NHANES)', hint: 'EBM, CatBoost, XGBoost and LightGBM compared' },
];

export function RiskPredictionPage() {
  const { patientId } = useParams<{ patientId: string }>();
  const { hospitalId } = useHospital();
  const [mode, setMode] = useState<RiskMode>('urine');
  const parsedPatientId = Number(patientId);
  return (
    <AppLayout
      role="hospital"
      title="Clinical Risk Prediction"
      subtitle="Review clinical-risk evidence and explanations independently from imaging."
    >
      {Number.isInteger(parsedPatientId) ? (
        <>
          <div role="tablist" aria-label="Risk model input" className="mb-5 inline-flex rounded-lg border border-stonesense-line bg-white p-1 shadow-sm">
            {modes.map((m) => (
              <button
                key={m.key}
                role="tab"
                type="button"
                aria-selected={mode === m.key}
                onClick={() => setMode(m.key)}
                title={m.hint}
                className={`rounded-md px-4 py-1.5 text-sm font-medium transition ${
                  mode === m.key ? 'bg-stonesense-teal text-white' : 'text-stonesense-ink/70 hover:bg-black/5'
                }`}
              >
                {m.label}
              </button>
            ))}
          </div>
          {mode === 'urine' ? (
            <PatientForm patientId={parsedPatientId} />
          ) : (
            <NhanesClinicalProfile patientId={parsedPatientId} hospitalId={hospitalId ?? 1} />
          )}
        </>
      ) : (
        <p className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-700">A valid patient is required.</p>
      )}
    </AppLayout>
  );
}
