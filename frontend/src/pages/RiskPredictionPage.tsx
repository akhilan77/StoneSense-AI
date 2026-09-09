import AppLayout from '../components/layout/AppLayout';
import { PatientForm } from '../components/PatientForm';

export function RiskPredictionPage() {
  return (
    <AppLayout
      role="hospital"
      title="Clinical Risk Prediction"
      subtitle="Review XGBoost clinical-risk evidence and SHAP explanation independently from imaging."
    >
      <PatientForm />
    </AppLayout>
  );
}
