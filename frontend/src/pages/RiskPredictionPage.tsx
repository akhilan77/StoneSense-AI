import AppLayout from '../components/layout/AppLayout';
import { PatientForm } from '../components/PatientForm';
import { useParams } from 'react-router-dom';

export function RiskPredictionPage() {
  const { patientId } = useParams<{ patientId: string }>();
  const parsedPatientId = Number(patientId);
  return (
    <AppLayout
      role="hospital"
      title="Clinical Risk Prediction"
      subtitle="Review XGBoost clinical-risk evidence and SHAP explanation independently from imaging."
    >
      {Number.isInteger(parsedPatientId) ? (
        <PatientForm patientId={parsedPatientId} />
      ) : (
        <p className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-700">A valid patient is required.</p>
      )}
    </AppLayout>
  );
}
