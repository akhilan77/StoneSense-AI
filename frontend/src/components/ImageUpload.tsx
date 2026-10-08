import { useEffect, useMemo, useState } from 'react';
import { API_BASE_URL, fetchDLModels, fetchPatient, predictImage } from '../services/api';
import type { DLModelMetadata, StoneDetection } from '../types/stoneDetection';
import { ErrorMessage } from './ErrorMessage';
import { LoadingSpinner } from './LoadingSpinner';
import { PredictionCard } from './PredictionCard';
import { useHospital } from '../context/HospitalContext';
import type { PatientRecord } from '../types/dashboard';

const acceptedMimeTypes = ['image/jpeg', 'image/png', 'image/jpg'];

const DEFAULT_MODELS: DLModelMetadata[] = [
  { id: 'resnet18', name: 'ResNet18', family: 'resnet18_ct', status: 'READY', is_ready: true, accuracy: 0.9963, macro_f1: 0.9947 },
  { id: 'yolo26', name: 'YOLO26', family: 'yolo26_ct', status: 'PENDING_WEIGHTS', is_ready: false, accuracy: 0.9219, macro_f1: 0.9047 },
  { id: 'dinov3', name: 'DINOv3', family: 'dinov3_ct', status: 'PENDING_WEIGHTS', is_ready: false },
  { id: 'qknn', name: 'QKNN', family: 'qknn_ct', status: 'PENDING_WEIGHTS', is_ready: false },
];

export function ImageUpload({ patientId }: { patientId: number }) {
  const { hospitalId } = useHospital();
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [detection, setDetection] = useState<StoneDetection | null>(null);
  const [patientRecord, setPatientRecord] = useState<PatientRecord | null>(null);
  const [isPatientLoading, setIsPatientLoading] = useState(true);

  // Multi-Model Selection States
  const [models, setModels] = useState<DLModelMetadata[]>(DEFAULT_MODELS);
  const [selectedModelId, setSelectedModelId] = useState<string>('resnet18');

  useEffect(() => {
    setIsPatientLoading(true);
    fetchPatient(patientId, hospitalId ?? 1)
      .then(setPatientRecord)
      .catch((loadError) => setError(loadError instanceof Error ? loadError.message : 'Unable to load patient.'))
      .finally(() => setIsPatientLoading(false));

    fetchDLModels()
      .then((res) => {
        if (res.models && res.models.length > 0) {
          setModels(res.models);
          // If current selected model is not ready, pick first ready model
          const current = res.models.find((m) => m.id === selectedModelId);
          if (!current || !current.is_ready) {
            const firstReady = res.models.find((m) => m.is_ready);
            if (firstReady) setSelectedModelId(firstReady.id);
          }
        }
      })
      .catch(() => {
        // Retain default models on failure
      });
  }, [patientId, hospitalId]);

  const activeModelMeta = useMemo(() => {
    return models.find((m) => m.id === selectedModelId) ?? models[0];
  }, [models, selectedModelId]);

  const detectionData = useMemo(
    () => [
      { label: 'Prediction', value: detection?.class_name ?? '—' },
      {
        label: 'Model Confidence',
        value: detection ? `${((detection?.confidence ?? 0) * 100).toFixed(1)}%` : '—',
      },
      { label: 'Model', value: detection?.model_name ?? activeModelMeta?.name ?? 'ResNet18' },
    ],
    [detection, activeModelMeta]
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

    setDetection(null);
    setIsLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('image', imageFile);
    formData.append('model_family', selectedModelId);
    formData.append('hospital_id', String(hospitalId ?? 1));
    formData.append('patient_id', String(patientId));

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
    <div>
      {isPatientLoading ? (
        <div className="mb-6 rounded-lg border border-dashed border-stonesense-line bg-white p-5 text-sm text-stonesense-ink/60">Loading patient...</div>
      ) : patientRecord ? (
        <section className="mb-6 rounded-lg border border-stonesense-line bg-white p-5 shadow-sm">
          <p className="text-xs font-semibold uppercase tracking-wider text-stonesense-teal">Patient</p>
          <div className="mt-2 grid gap-3 text-sm sm:grid-cols-2"><div><p className="font-serif text-lg text-stonesense-ink">{patientRecord.name}</p><p className="text-stonesense-ink/60">{patientRecord.phone}</p></div><div className="grid grid-cols-2 gap-2 text-xs text-stonesense-ink/70"><p><strong>Patient ID:</strong> {patientRecord.patient_id}</p><p><strong>Blood Group:</strong> {patientRecord.blood_group}</p><p><strong>Date of Admission:</strong> {patientRecord.admitted_date}</p><p><strong>Last Inspected:</strong> {patientRecord.last_inspected_date ?? '—'}</p></div></div>
        </section>
      ) : null}

      {/* Model Selection Header */}
      <section className="mb-6 rounded-lg border border-stonesense-line bg-white p-5 shadow-sm">
        <div className="flex flex-col gap-1 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <p className="text-xs font-semibold uppercase tracking-wider text-stonesense-teal">Select DL Model</p>
            <h2 className="font-serif text-lg text-stonesense-ink">Classification Architecture</h2>
          </div>
          <p className="text-xs text-stonesense-ink/60">Choose from 4 supported CT imaging models</p>
        </div>

        <div className="mt-4 grid grid-cols-2 gap-3 md:grid-cols-4">
          {models.map((m) => {
            const isSelected = m.id === selectedModelId;
            const isReady = m.is_ready || m.status === 'READY';
            return (
              <button
                key={m.id}
                type="button"
                disabled={!isReady}
                onClick={() => isReady && setSelectedModelId(m.id)}
                className={`relative flex flex-col rounded-lg border p-3.5 text-left transition-all ${
                  isSelected
                    ? 'border-stonesense-teal bg-stonesense-teal/5 shadow-sm ring-1 ring-stonesense-teal'
                    : isReady
                    ? 'border-stonesense-line bg-white hover:border-stonesense-teal/50 hover:bg-stonesense-paper'
                    : 'cursor-not-allowed border-dashed border-stonesense-line bg-stone-50 opacity-60'
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className="font-serif font-semibold text-stonesense-ink">{m.name}</span>
                  <span
                    className={`rounded-full px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider ${
                      isReady
                        ? 'bg-emerald-100 text-emerald-800'
                        : m.status === 'PENDING_WEIGHTS'
                        ? 'bg-amber-100 text-amber-800'
                        : 'bg-red-100 text-red-800'
                    }`}
                  >
                    {isReady ? 'Ready' : 'Pending Weights'}
                  </span>
                </div>

                <div className="mt-2 text-xs text-stonesense-ink/70">
                  {m.accuracy != null ? (
                    <p>Accuracy: <strong>{(m.accuracy * 100).toFixed(1)}%</strong></p>
                  ) : (
                    <p className="italic text-stonesense-ink/40">Weights pending export</p>
                  )}
                  {m.macro_f1 != null ? (
                    <p className="text-[11px] text-stonesense-ink/50">Macro F1: {(m.macro_f1 * 100).toFixed(1)}%</p>
                  ) : null}
                </div>
              </button>
            );
          })}
        </div>
      </section>

      <div className="grid gap-6 xl:grid-cols-[1.1fr_0.9fr]">
        <form
          onSubmit={handleSubmit}
          className="rounded-lg border border-stonesense-line bg-white p-6 shadow-sm"
        >
          <div className="mb-4">
            <h2 className="font-serif text-lg text-stonesense-ink">CT Image Input</h2>
            <p className="text-sm text-stonesense-ink/60 mt-0.5">
              Upload a renal CT slice to run inference using <strong>{activeModelMeta?.name}</strong>.
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
            disabled={isLoading || !imageFile || !activeModelMeta?.is_ready}
            className="mt-4 rounded-md bg-stonesense-teal px-4 py-2 text-sm font-medium text-white transition hover:bg-stonesense-teal/90 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {isLoading ? 'Analyzing...' : `Run ${activeModelMeta?.name} assessment`}
          </button>

          {isLoading ? (
            <div className="mt-4">
              <LoadingSpinner label={`Running ${activeModelMeta?.name} classification...`} />
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
                <h2 className="mt-1 font-serif text-lg text-stonesense-ink">Diagnostic Assessment</h2>
                <h3 className="mt-3 font-serif text-base text-stonesense-ink">
                  {detection.gradcam?.available ? 'Grad-CAM Visual Evidence' : 'Explainability Information'}
                </h3>
                <p className="mt-2 text-sm text-stonesense-ink/60">
                  {detection.gradcam?.available
                    ? 'The highlighted regions indicate areas that contributed to the model\'s prediction. This visualization is an AI explanation and does not by itself establish a clinical diagnosis.'
                    : detection.gradcam?.message ||
                      (detection.class_name === 'Normal'
                        ? 'No stone-specific localization is shown because the model classified this scan as Normal.'
                        : `Explainability visualization is not supported for ${detection.model_name || 'this model'}.`)}
                </p>
                {detection.gradcam?.overlay_url && detection.gradcam.available ? (
                  <img
                    src={`${API_BASE_URL}${detection.gradcam.overlay_url}`}
                    alt="Grad-CAM explanation overlay"
                    className="mt-4 max-h-72 w-full rounded-lg border border-stonesense-line object-contain"
                    onError={(event) => {
                      event.currentTarget.style.display = 'none';
                    }}
                  />
                ) : (
                  <div className="mt-4 rounded-lg border border-dashed border-stonesense-line bg-stonesense-paper p-6 text-center text-sm text-stonesense-ink/60">
                    {detection.gradcam?.message || 'Explanation visualization unavailable for this scan.'}
                  </div>
                )}
                <div className="mt-5 grid gap-3 border-t border-stonesense-line pt-4 sm:grid-cols-2">
                  <div>
                    <p className="text-xs text-stonesense-ink/50">Model</p>
                    <p className="font-semibold text-stonesense-ink">{detection.model_name ?? activeModelMeta?.name ?? 'ResNet18'}</p>
                  </div>
                  <div>
                    <p className="text-xs text-stonesense-ink/50">Explainability</p>
                    <p className="font-semibold text-stonesense-ink">
                      {detection.gradcam?.available ? 'Grad-CAM' : 'Not Supported'}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-stonesense-ink/50">Input</p>
                    <p className="font-semibold text-stonesense-ink">CT Image</p>
                  </div>
                  <div>
                    <p className="text-xs text-stonesense-ink/50">Target class</p>
                    <p className="font-semibold text-stonesense-ink">{detection.gradcam?.target_class ?? detection.class_name}</p>
                  </div>
                </div>
              </section>
            </div>
          ) : (
            <div className="rounded-lg border border-dashed border-stonesense-line bg-white p-6 text-center text-sm text-stonesense-ink/50">
              Upload an image and select a model to inspect the CT classification result.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
