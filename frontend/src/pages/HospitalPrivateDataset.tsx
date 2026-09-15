import { ChangeEvent, useEffect, useRef, useState } from 'react';
import AppLayout from '../components/layout/AppLayout';
import { useHospital } from '../context/HospitalContext';
import {
  fetchHospitalDatasetStatus,
  uploadHospitalDataset,
  validateHospitalDataset,
} from '../services/federatedApi';
import { DatasetStatus } from '../types/federated';

const classes = ['Cyst', 'Normal', 'Stone', 'Tumor'] as const;

export default function HospitalPrivateDataset() {
  const { hospitalId, hospitals } = useHospital();
  const activeHospitalId = hospitalId ?? 1;
  const hospital = hospitals.find((item) => item.id === activeHospitalId);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [dataset, setDataset] = useState<DatasetStatus | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(true);
  const [working, setWorking] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);

  const loadDataset = async () => {
    setLoading(true);
    try {
      setDataset(await fetchHospitalDatasetStatus(activeHospitalId));
      setError(null);
    } catch (loadError) {
      setError(
        loadError instanceof Error ? loadError.message : 'Unable to load private dataset status.'
      );
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadDataset();
  }, [activeHospitalId]);

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0] ?? null;
    setSelectedFile(file);
    setMessage(file ? `${file.name} selected.` : null);
    setError(null);
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setError('Choose a ZIP dataset before uploading.');
      return;
    }
    setWorking(true);
    setError(null);
    setMessage('Validating and installing the private dataset...');
    try {
      const nextDataset = await uploadHospitalDataset(activeHospitalId, selectedFile);
      setDataset(nextDataset);
      setSelectedFile(null);
      if (fileInputRef.current) fileInputRef.current.value = '';
      setMessage('Private dataset uploaded and validated successfully.');
    } catch (uploadError: any) {
      setError(uploadError.response?.data?.detail ?? 'Private dataset upload failed.');
      setMessage(null);
    } finally {
      setWorking(false);
    }
  };

  const handleValidate = async () => {
    setWorking(true);
    setError(null);
    try {
      const result = await validateHospitalDataset(activeHospitalId);
      setMessage(result.message);
      await loadDataset();
    } catch (validationError: any) {
      setError(validationError.response?.data?.detail ?? 'Private dataset validation failed.');
    } finally {
      setWorking(false);
    }
  };

  return (
    <AppLayout
      role="hospital"
      title="Private Dataset"
      subtitle="Manage this hospital's private CT dataset for local DL training."
    >
      <div className="space-y-6">
        {error && (
          <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
            {error}
          </div>
        )}
        {message && (
          <div className="rounded-lg border border-[#D2E0D1] bg-[#EBF5F1] px-4 py-3 text-sm text-[#1F6F5C]">
            {message}
          </div>
        )}

        <section className="rounded-xl border border-[#DDE3DC] bg-white p-6 shadow-xs">
          <div className="flex flex-col gap-4 border-b border-[#DDE3DC] pb-5 md:flex-row md:items-start md:justify-between">
            <div>
              <div className="mb-2 flex items-center gap-2">
                <span
                  className={`h-2 w-2 rounded-full ${dataset?.is_valid ? 'bg-[#1F6F5C]' : 'bg-[#C97A2B]'}`}
                />
                <span className="text-xs font-semibold uppercase tracking-wider text-[#101B16]/55">
                  Hospital-private storage
                </span>
              </div>
              <h2 className="font-serif text-2xl font-medium text-[#101B16]">
                {hospital?.name ?? 'Hospital dataset'}
              </h2>
              <p className="mt-1 font-mono text-xs text-[#101B16]/55">
                {hospital?.hospital_code ?? `HOSP-${String(activeHospitalId).padStart(3, '0')}`}
              </p>
            </div>
            <div className="text-left md:text-right">
              <p className="text-[11px] font-semibold uppercase tracking-wider text-[#101B16]/50">
                Status
              </p>
              <p
                className={`mt-1 text-sm font-semibold ${dataset?.is_valid ? 'text-[#1F6F5C]' : 'text-[#C97A2B]'}`}
              >
                {loading ? 'Loading...' : dataset?.is_valid ? 'Ready' : 'Needs validation'}
              </p>
            </div>
          </div>

          <div className="mt-5 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <Metric
              label="Dataset version"
              value={dataset?.dataset_version ?? 'Not uploaded'}
              mono
            />
            <Metric label="Samples" value={dataset ? dataset.dataset_size.toLocaleString() : '-'} />
            <Metric label="Validation" value={dataset?.is_valid ? 'Passed' : 'Not passed'} />
            <Metric label="Last updated" value={formatDate(dataset?.last_updated)} />
          </div>
        </section>

        <section className="grid gap-6 lg:grid-cols-[1.15fr_0.85fr]">
          <div className="rounded-xl border border-[#DDE3DC] bg-white p-6 shadow-xs">
            <h3 className="text-sm font-semibold text-[#101B16]">Upload or replace dataset</h3>
            <p className="mt-1 text-xs leading-5 text-[#101B16]/60">
              Upload a ZIP containing train, validation, and test folders. Each split must contain
              Cyst, Normal, Stone, and Tumor class folders with supported image files.
            </p>
            <input
              ref={fileInputRef}
              type="file"
              accept=".zip,application/zip"
              onChange={handleFileChange}
              className="mt-5 block w-full rounded-lg border border-[#DDE3DC] bg-[#F7F9F6] px-3 py-3 text-xs text-[#101B16] file:mr-3 file:rounded-md file:border-0 file:bg-[#1F6F5C] file:px-3 file:py-2 file:text-xs file:font-semibold file:text-white"
            />
            <div className="mt-4 flex flex-wrap gap-2">
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                className="rounded-lg border border-[#DDE3DC] bg-white px-4 py-2 text-xs font-semibold text-[#101B16] hover:bg-[#F7F9F6]"
              >
                Choose ZIP
              </button>
              <button
                type="button"
                onClick={handleUpload}
                disabled={!selectedFile || working}
                className="rounded-lg bg-[#1F6F5C] px-4 py-2 text-xs font-semibold text-white hover:bg-[#185849] disabled:cursor-not-allowed disabled:opacity-50"
              >
                {working ? 'Processing...' : 'Upload Dataset'}
              </button>
              <button
                type="button"
                onClick={handleValidate}
                disabled={working || !dataset}
                className="rounded-lg border border-[#DDE3DC] bg-white px-4 py-2 text-xs font-semibold text-[#101B16] hover:bg-[#F7F9F6] disabled:cursor-not-allowed disabled:opacity-50"
              >
                Validate Dataset
              </button>
            </div>
            <p className="mt-4 text-[11px] text-[#101B16]/50">
              Raw patient and CT files remain in this hospital's local dataset directory. Only model
              parameters and aggregate telemetry enter DL federated training.
            </p>
          </div>

          <div className="rounded-xl border border-[#DDE3DC] bg-white p-6 shadow-xs">
            <h3 className="text-sm font-semibold text-[#101B16]">Class distribution</h3>
            <div className="mt-4 space-y-3">
              {classes.map((className) => (
                <div
                  key={className}
                  className="flex items-center justify-between rounded-lg bg-[#F7F9F6] px-3 py-2.5 text-xs"
                >
                  <span className="font-medium text-[#101B16]">{className}</span>
                  <strong className="font-mono text-[#1F6F5C]">
                    {dataset?.class_distribution?.[className]?.toLocaleString() ?? '0'}
                  </strong>
                </div>
              ))}
            </div>
            <div className="mt-5 border-t border-[#DDE3DC] pt-4">
              <h4 className="text-[11px] font-semibold uppercase tracking-wider text-[#101B16]/50">
                Split counts
              </h4>
              <div className="mt-3 grid grid-cols-3 gap-2">
                {(['train', 'validation', 'test'] as const).map((split) => (
                  <Metric
                    key={split}
                    label={split}
                    value={String(dataset?.split_info?.[split] ?? 0)}
                  />
                ))}
              </div>
            </div>
          </div>
        </section>
      </div>
    </AppLayout>
  );
}

function formatDate(value?: string) {
  return value ? new Date(value).toLocaleString() : '-';
}

function Metric({ label, value, mono = false }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="rounded-lg border border-[#DDE3DC] bg-[#F7F9F6] p-3">
      <p className="text-[10px] font-semibold uppercase tracking-wider text-[#101B16]/50">
        {label}
      </p>
      <p
        className={`mt-1 truncate text-sm font-semibold text-[#101B16] ${mono ? 'font-mono text-xs' : ''}`}
      >
        {value}
      </p>
    </div>
  );
}
