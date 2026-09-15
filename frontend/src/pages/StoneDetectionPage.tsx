import AppLayout from '../components/layout/AppLayout';
import { ImageUpload } from '../components/ImageUpload';
import { useParams } from 'react-router-dom';

export function StoneDetectionPage() {
  const { patientId } = useParams<{ patientId: string }>();
  const parsedPatientId = Number(patientId);
  return (
    <AppLayout
      role="hospital"
      title="CT Stone Detection"
      subtitle="Review ResNet18 imaging evidence and Grad-CAM explanation independently from clinical risk."
    >
      {Number.isInteger(parsedPatientId) ? (
        <ImageUpload patientId={parsedPatientId} />
      ) : (
        <p className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-700">A valid patient is required.</p>
      )}
    </AppLayout>
  );
}
