import axios from 'axios';
import { useEffect, useMemo, useState } from 'react';
import { fetchNhanesSpec, predictRiskCompare } from '../services/api';
import type {
  NhanesCompareResponse,
  NhanesFeatureSpec,
  NhanesFeatureSpecResponse,
  NhanesFeatureValues,
} from '../types/riskPrediction';
import { ErrorMessage } from './ErrorMessage';
import { LoadingSpinner } from './LoadingSpinner';
import { NhanesComparisonResults } from './NhanesComparisonResults';

const fieldClassName =
  'mt-1 w-full rounded-md border border-stonesense-line bg-white px-3 py-1.5 text-sm text-stonesense-ink outline-none transition focus:border-stonesense-teal';

function describeError(error: unknown): string {
  if (axios.isAxiosError(error)) {
    const detail = error.response?.data?.detail;
    if (detail && typeof detail === 'object' && Array.isArray(detail.problems)) {
      return `${detail.message ?? 'Invalid input.'} ${detail.problems.join('; ')}`;
    }
    if (typeof detail === 'string') return detail;
  }
  return error instanceof Error ? error.message : 'Unable to fetch the model comparison.';
}

interface Props {
  patientId?: number;
  hospitalId: number;
  /** Called with the latest comparison (or null when a new run starts). */
  onResult?: (result: NhanesCompareResponse | null) => void;
  /** Render the result table below the form (default true). The dashboard renders it in its results step instead. */
  showResults?: boolean;
  /** Seed values keyed by NHANES variable name (e.g. from a stored patient profile). */
  initialValues?: Record<string, number | null | undefined>;
  submitLabel?: string;
}

export function NhanesClinicalProfile({
  patientId,
  hospitalId,
  onResult,
  showResults = true,
  initialValues,
  submitLabel = 'Run four-model assessment',
}: Props) {
  const [spec, setSpec] = useState<NhanesFeatureSpecResponse | null>(null);
  const [values, setValues] = useState<Record<string, string>>({});
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<NhanesCompareResponse | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetchNhanesSpec()
      .then((s) => {
        if (cancelled) return;
        setSpec(s);
        const seeded: Record<string, string> = {};
        for (const f of s.features) {
          if (f.type === 'derived') continue;
          const seed = initialValues?.[f.name];
          const v = seed !== undefined ? seed : f.default;
          seeded[f.name] = v === null || v === undefined ? '' : String(v);
        }
        setValues(seeded);
      })
      .catch((e) => !cancelled && setError(describeError(e)));
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const groups = useMemo(() => {
    const map = new Map<string, NhanesFeatureSpec[]>();
    for (const f of spec?.features ?? []) {
      if (f.type === 'derived') continue;
      const key = f.is_history ? 'Medical history' : f.group;
      map.set(key, [...(map.get(key) ?? []), f]);
    }
    return Array.from(map.entries());
  }, [spec]);

  const derived = (spec?.features ?? []).filter((f) => f.type === 'derived');

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!spec) return;
    setError(null);
    const payload: NhanesFeatureValues = {};
    const problems: string[] = [];
    for (const f of spec.features) {
      if (f.type === 'derived') continue;
      const raw = values[f.name];
      if (raw === undefined || raw === '') {
        if (f.required) problems.push(`${f.label} is required`);
        payload[f.name] = null;
        continue;
      }
      const n = Number(raw);
      if (!Number.isFinite(n)) {
        problems.push(`${f.label} must be a number`);
      } else if (!f.options && (n < f.min || n > f.max)) {
        problems.push(`${f.label} must be between ${f.min} and ${f.max} ${f.unit}`.trim());
      }
      payload[f.name] = n;
    }
    if (problems.length > 0) {
      setError(problems.join('; '));
      return;
    }
    setIsLoading(true);
    setResult(null);
    onResult?.(null);
    try {
      const response = await predictRiskCompare({ features: payload, hospital_id: hospitalId, patient_id: patientId });
      setResult(response);
      onResult?.(response);
    } catch (submitError) {
      setError(describeError(submitError));
    } finally {
      setIsLoading(false);
    }
  };

  const renderField = (f: NhanesFeatureSpec) => {
    const labelText = f.is_history && f.question ? f.question : `${f.label}${f.unit ? ` (${f.unit})` : ''}`;
    const id = `nhanes-${f.name}`;
    return (
      <label key={f.name} htmlFor={id} className={`text-xs text-stonesense-ink/60 ${f.is_history ? 'sm:col-span-2' : ''}`}>
        {labelText}
        {f.options ? (
          <select
            id={id}
            className={fieldClassName}
            value={values[f.name] ?? ''}
            onChange={(e) => setValues((cur) => ({ ...cur, [f.name]: e.target.value }))}
          >
            {!f.required ? <option value="">{f.sensitive ? 'Prefer not to say' : 'Not sure / skip'}</option> : null}
            {f.options.map((o) => (
              <option key={o.value} value={String(o.value)}>
                {o.label}
              </option>
            ))}
          </select>
        ) : (
          <input
            id={id}
            type="number"
            step="any"
            min={f.min}
            max={f.max}
            className={fieldClassName}
            value={values[f.name] ?? ''}
            placeholder={f.required ? '' : 'optional'}
            onChange={(e) => setValues((cur) => ({ ...cur, [f.name]: e.target.value }))}
          />
        )}
      </label>
    );
  };

  return (
    <div className="space-y-6">
      <form onSubmit={handleSubmit} noValidate className="rounded-lg border border-stonesense-line bg-white p-6 shadow-sm">
        <div className="mb-5 flex items-center justify-between gap-3">
          <div>
            <h2 className="font-serif text-lg text-stonesense-ink">Clinical profile (NHANES)</h2>
            <p className="mt-0.5 text-sm text-stonesense-ink/60">
              Blood chemistry, urine, body measures and self-reported history, scored by four models trained on NHANES 2007–2018. Defaults are
              training-cohort medians; replace them with this patient&apos;s values.
            </p>
          </div>
          <button
            type="submit"
            disabled={isLoading || !spec}
            className="rounded-md bg-stonesense-teal px-4 py-2 text-sm font-medium text-white transition hover:bg-stonesense-teal/90 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {isLoading ? 'Scoring...' : submitLabel}
          </button>
        </div>

        {!spec && !error ? <LoadingSpinner label="Loading the clinical-profile form..." /> : null}

        {groups.map(([group, feats]) => {
          const sensitive = feats.every((f) => f.sensitive);
          const fields = <div className="grid gap-4 sm:grid-cols-2">{feats.map(renderField)}</div>;
          return (
            <fieldset key={group} className="mb-5">
              {sensitive ? (
                <details>
                  <summary className="cursor-pointer text-xs font-semibold uppercase tracking-wider text-stonesense-ink/60">
                    {group} inputs (optional)
                  </summary>
                  <p className="mb-3 mt-2 text-xs text-stonesense-ink/60">
                    Race / ethnicity and income are sensitive. The models can use them, but they can be left blank; blank values are replaced by the
                    training-cohort typical value.
                  </p>
                  {fields}
                </details>
              ) : (
                <>
                  <legend className="mb-2 text-xs font-semibold uppercase tracking-wider text-stonesense-ink/60">{group}</legend>
                  {fields}
                </>
              )}
            </fieldset>
          );
        })}

        {derived.length > 0 ? (
          <p className="text-xs text-stonesense-ink/50">
            {derived.map((d) => `${d.label} is calculated by the server (${d.formula})`).join('; ')}.
          </p>
        ) : null}

        {isLoading ? (
          <div className="mt-4">
            <LoadingSpinner label="Running the four models..." />
          </div>
        ) : null}
        {error ? (
          <div className="mt-4">
            <ErrorMessage message={error} />
          </div>
        ) : null}
      </form>

      {showResults && result ? <NhanesComparisonResults result={result} /> : null}
      {showResults && !result && !isLoading ? (
        <div className="rounded-lg border border-dashed border-stonesense-line bg-white p-6 text-center text-sm text-stonesense-ink/50">
          Run the assessment to compare the EBM, CatBoost, XGBoost and LightGBM models.
        </div>
      ) : null}
    </div>
  );
}
