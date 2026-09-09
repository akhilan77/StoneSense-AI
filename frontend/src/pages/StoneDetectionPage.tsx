import AppLayout from '../components/layout/AppLayout';
import { ImageUpload } from '../components/ImageUpload';

export function StoneDetectionPage() {
  return (
    <AppLayout
      role="hospital"
      title="CT Stone Detection"
      subtitle="Review ResNet18 imaging evidence and Grad-CAM explanation independently from clinical risk."
    >
      <ImageUpload />
    </AppLayout>
  );
}
