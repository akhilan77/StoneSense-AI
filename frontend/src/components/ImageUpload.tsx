import { useMemo, useState } from 'react';
import { API_BASE_URL, predictImage } from '../services/api';
import type { StoneDetection } from '../types/stoneDetection';
import { ErrorMessage } from './ErrorMessage';
import { LoadingSpinner } from './LoadingSpinner';
import { PredictionCard } from './PredictionCard';
import { useHospital } from '../context/HospitalContext';

const acceptedMimeTypes = ['image/jpeg', 'image/png', 'image/jpg'];

export function ImageUpload() {
  const { hospitalId } = useHospital();
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [detection, setDetection] = useState<StoneDetection | null>(null);

  const detectionData = useMemo(
    () => [
      { label: 'Prediction', value: detection?.class_name ?? '—' },
      {
        label: 'Model Confidence',
        value: detection ? `${((detection?.confidence ?? 0) * 100).toFixed(1)}%` : '—',
      },
      { label: 'Model', value: 'ResNet18' },
    ],
    [detection]
  );

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

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    if (!imageFile) {
      setError('Please select an image before uploading.');
      return;
    }

    setIsLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('image', imageFile);
    formData.append('hospital_id', String(hospitalId ?? 1));

    try {
      const response = await predictImage(formData);
      setDetection(response);
    } catch (submitError) {
      setError(
        submitError instanceof Error ? submitError.message : 'Unable to run image detection.'
      );
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="grid gap-6 xl:grid-cols-[1.1fr_0.9fr]">
      <form
        onSubmit={handleSubmit}
        className="rounded-lg border border-stonesense-line bg-white p-6 shadow-sm"
      >
        <div className="mb-4">
          <h2 className="font-serif text-lg text-stonesense-ink">CT Image Input</h2>
          <p className="text-sm text-stonesense-ink/60 mt-0.5">
            Upload a renal image to request an AI classification result.
          </p>
        </div>

        <label className="flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed border-stonesense-line bg-stonesense-paper p-8 text-center hover:border-stonesense-teal/50 transition-colors">
          <span className="text-sm font-medium text-stonesense-teal">Choose image or drop CT slice here</span>
          <span className="mt-1 text-xs text-stonesense-ink/50">Allowed: DICOM / PNG / JPG / JPEG</span>
          <input
            type="file"
            accept=".jpg,.jpeg,.png,image/jpeg,image/png"
            className="hidden"
            onChange={handleFileChange}
          />
        </label>

        {previewUrl ? (
          <div className="mt-4 overflow-hidden rounded-lg border border-stonesense-line">
            <img src={previewUrl} alt="Uploaded preview" className="max-h-72 w-full object-cover" />
          </div>
        ) : null}

        <button
          type="submit"
          disabled={isLoading || !imageFile}
          className="mt-4 rounded-md bg-stonesense-teal px-4 py-2 text-sm font-medium text-white transition hover:bg-stonesense-teal/90 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {isLoading ? 'Uploading...' : 'Run assessment'}
        </button>

        {isLoading ? (
          <div className="mt-4">
            <LoadingSpinner label="Analyzing uploaded image..." />
          </div>
        ) : null}
        {error ? (
          <div className="mt-4">
            <ErrorMessage message={error} />
          </div>
        ) : null}
      </form>

      <div className="space-y-4">
        {detection ? (
          <div className="space-y-4">
            <PredictionCard title="Imaging Assessment" data={detectionData} />
            <section className="rounded-lg border border-stonesense-line bg-white p-6 shadow-sm">
              <p className="text-xs font-semibold uppercase tracking-wider text-stonesense-teal">
                Imaging Evidence
              </p>
              <h2 className="mt-1 font-serif text-lg text-stonesense-ink">Trustworthy Assessment</h2>
              <h3 className="mt-3 font-serif text-base text-stonesense-ink">Grad-CAM Explanation</h3>
              <p className="mt-2 text-sm text-stonesense-ink/60">
                The highlighted region influenced the model prediction; it does not by itself prove
                the presence of a stone.
              </p>
              {detection.gradcam?.overlay_url ? (
                <img
                  src={`${API_BASE_URL}${detection.gradcam.overlay_url}`}
                  alt="Grad-CAM explanation overlay"
                  className="mt-4 max-h-72 w-full rounded-lg border border-stonesense-line object-contain"
                />
              ) : null}
              <div className="mt-5 grid gap-3 border-t border-stonesense-line pt-4 sm:grid-cols-2">
                <div>
                  <p className="text-xs text-stonesense-ink/50">Model</p>
                  <p className="font-semibold text-stonesense-ink">ResNet18</p>
                </div>
                <div>
                  <p className="text-xs text-stonesense-ink/50">Explainability</p>
                  <p className="font-semibold text-stonesense-ink">Grad-CAM</p>
                </div>
                <div>
                  <p className="text-xs text-stonesense-ink/50">Input</p>
                  <p className="font-semibold text-stonesense-ink">CT Image</p>
                </div>
                <div>
                  <p className="text-xs text-stonesense-ink/50">Classes</p>
                  <p className="font-semibold text-stonesense-ink">Cyst / Normal / Stone / Tumor</p>
                </div>
              </div>
            </section>
          </div>
        ) : (
          <div className="rounded-lg border border-dashed border-stonesense-line bg-white p-6 text-center text-sm text-stonesense-ink/50">
            Upload an image to inspect the classification prediction result.
          </div>
        )}
      </div>
    </div>
  );
}
