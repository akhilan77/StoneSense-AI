import { useMemo, useState } from 'react';
import { API_BASE_URL, predictImage } from '../services/api';
import type { StoneDetection } from '../types/stoneDetection';
import { ErrorMessage } from './ErrorMessage';
import { LoadingSpinner } from './LoadingSpinner';
import { PredictionCard } from './PredictionCard';

const acceptedMimeTypes = ['image/jpeg', 'image/png', 'image/jpg'];

export function ImageUpload() {
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
        className="rounded-3xl border border-slate-800 bg-slate-900/70 p-5 shadow-xl shadow-slate-950/40"
      >
        <div className="mb-4">
          <h2 className="text-xl font-semibold text-white">CT Image Input</h2>
          <p className="text-sm text-slate-400">
            Upload a renal image to request an AI classification result.
          </p>
        </div>

        <label className="flex cursor-pointer flex-col items-center justify-center rounded-3xl border border-dashed border-cyan-400/40 bg-slate-950 p-6 text-center">
          <span className="text-sm font-medium text-cyan-300">Choose image</span>
          <span className="mt-1 text-xs text-slate-400">Allowed: jpg, jpeg, png</span>
          <input
            type="file"
            accept=".jpg,.jpeg,.png,image/jpeg,image/png"
            className="hidden"
            onChange={handleFileChange}
          />
        </label>

        {previewUrl ? (
          <div className="mt-4 overflow-hidden rounded-2xl border border-slate-800">
            <img src={previewUrl} alt="Uploaded preview" className="max-h-72 w-full object-cover" />
          </div>
        ) : null}

        <button
          type="submit"
          disabled={isLoading || !imageFile}
          className="mt-4 rounded-full bg-cyan-400 px-5 py-2 text-sm font-semibold text-slate-950 transition hover:bg-cyan-300 disabled:cursor-not-allowed disabled:bg-slate-700 disabled:text-slate-400"
        >
          {isLoading ? 'Uploading...' : 'Upload image'}
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
            <section className="rounded-3xl border border-cyan-400/30 bg-slate-900/70 p-5">
              <p className="text-xs font-bold uppercase tracking-[0.18em] text-cyan-300">
                Imaging Evidence
              </p>
              <h2 className="mt-2 text-xl font-semibold text-white">Trustworthy Assessment</h2>
              <h3 className="mt-4 text-sm font-semibold text-white">Grad-CAM Explanation</h3>
              <p className="mt-2 text-sm text-slate-400">
                The highlighted region influenced the model prediction; it does not by itself prove
                the presence of a stone.
              </p>
              {detection.gradcam?.overlay_url ? (
                <img
                  src={`${API_BASE_URL}${detection.gradcam.overlay_url}`}
                  alt="Grad-CAM explanation overlay"
                  className="mt-4 max-h-72 w-full rounded-2xl border border-slate-700 object-contain"
                />
              ) : null}
              <div className="mt-5 grid gap-3 border-t border-slate-800 pt-4 sm:grid-cols-2">
                <div>
                  <p className="text-xs text-slate-500">Model</p>
                  <p className="font-semibold text-white">ResNet18</p>
                </div>
                <div>
                  <p className="text-xs text-slate-500">Explainability</p>
                  <p className="font-semibold text-white">Grad-CAM</p>
                </div>
                <div>
                  <p className="text-xs text-slate-500">Input</p>
                  <p className="font-semibold text-white">CT Image</p>
                </div>
                <div>
                  <p className="text-xs text-slate-500">Classes</p>
                  <p className="font-semibold text-white">Cyst / Normal / Stone / Tumor</p>
                </div>
              </div>
            </section>
          </div>
        ) : (
          <div className="rounded-3xl border border-dashed border-slate-700 bg-slate-900/40 p-6 text-sm text-slate-400">
            Upload an image to inspect the classification prediction result.
          </div>
        )}
      </div>
    </div>
  );
}
