import { useState } from 'react';
import type {
  NhanesCompareResponse,
  NhanesModelResult,
  NhanesOperatingPoint,
  NhanesOperatingPointKey,
  NhanesOperatingPointMetrics,
} from '../types/riskPrediction';

const OPERATING_POINT_OPTIONS: { key: NhanesOperatingPointKey; label: string; hint: string }[] = [
  { key: 'accuracy', label: 'Accuracy-focused (>70% accuracy)', hint: 'highest sensitivity that keeps validation accuracy at or above 72%' },
  { key: 'balanced', label: 'Balanced (max balanced accuracy)', hint: 'cut-off that maximises balanced accuracy on validation' },
];

const pct = (x: number | null | undefined, digits = 1) => (x === null || x === undefined ? '—' : `${(x * 100).toFixed(digits)}%`);
const ciText = (v: number, ci: [number, number]) => `${pct(v)} (95% CI ${pct(ci[0])} to ${pct(ci[1])})`;

function MetricCell({ value, ci, strong }: { value: number | undefined; ci?: [number, number]; strong?: boolean }) {
  if (value === undefined) return <td className="bg-stonesense-paper px-3 py-2 text-right text-stonesense-ink/40">—</td>;
  return (
    <td
      className={`bg-stonesense-paper px-3 py-2 text-right tabular-nums ${strong ? 'font-semibold text-stonesense-ink' : ''}`}
      title={ci ? ciText(value, ci) : undefined}
    >
      {pct(value)}
    </td>
  );
}

function OperatingPointLine({ m, baseline }: { m: NhanesOperatingPointMetrics | null; baseline: number }) {
  if (!m) return <span className="text-stonesense-ink/50">not reachable on the validation cycle</span>;
  return (
    <span>
      cut-off {m.threshold.toFixed(3)}: accuracy <strong>{pct(m.accuracy)}</strong> (majority baseline {pct(baseline)}), sensitivity{' '}
      <strong>{pct(m.sensitivity)}</strong>, specificity <strong>{pct(m.specificity)}</strong>, balanced accuracy {pct(m.balanced_accuracy)}
      {m.positive_rate !== undefined ? `, flags ${pct(m.positive_rate)} of people` : ''}
    </span>
  );
}

const opOf = (m: NhanesModelResult, key: NhanesOperatingPointKey): NhanesOperatingPoint | undefined => m.operating_points?.[key];

export function NhanesComparisonResults({ result, initialPoint = 'accuracy' }: { result: NhanesCompareResponse; initialPoint?: NhanesOperatingPointKey }) {
  const [selectedId, setSelectedId] = useState<string>(result.models[0]?.model_id ?? '');
  const [point, setPoint] = useState<NhanesOperatingPointKey>(initialPoint);
  const selected: NhanesModelResult | undefined = result.models.find((m) => m.model_id === selectedId) ?? result.models[0];
  const testCycle = result.models[0]?.test_metrics.test_cycle ?? '2017-2018';
  const baseline = result.models[0]?.test_metrics.majority_baseline_accuracy ?? 0;
  const riskAt = (m: NhanesModelResult) => opOf(m, point)?.risk_level ?? m.risk_level;
  const thresholdAt = (m: NhanesModelResult) => opOf(m, point)?.threshold ?? m.threshold;
  const nHigh = result.models.filter((m) => riskAt(m) === 'High').length;
  const agree = result.models.every((m) => riskAt(m) === riskAt(result.models[0]));
  const sensRange = result.models
    .map((m) => opOf(m, 'accuracy')?.test?.sensitivity)
    .filter((x): x is number => x !== undefined);
  const maxContribution = Math.max(0.001, ...(selected?.top_factors ?? []).map((f) => Math.abs(f.contribution)));

  return (
    <section className="space-y-4" aria-label="NHANES four-model comparison">
      <div className="rounded-lg border border-stonesense-line bg-white p-6 shadow-sm">
        <p className="text-xs font-semibold uppercase tracking-wider text-stonesense-teal">Clinical profile (NHANES)</p>
        <h2 className="mt-1 font-serif text-lg text-stonesense-ink">Four-model risk comparison</h2>

        <div className="mt-4 grid gap-3 sm:grid-cols-3">
          <div className="rounded-md border border-stonesense-line bg-stonesense-paper p-3.5">
            <div className="text-xs text-stonesense-ink/50">Average predicted probability</div>
            <div className="mt-1 text-xl font-semibold text-stonesense-ink">{pct(result.average_probability)}</div>
            <div className="text-xs text-stonesense-ink/50">
              range {pct(result.probability_min)} to {pct(result.probability_max)}
            </div>
          </div>
          <div className="rounded-md border border-stonesense-line bg-stonesense-paper p-3.5">
            <div className="text-xs text-stonesense-ink/50">Models flagging High risk</div>
            <div className="mt-1 text-xl font-semibold text-stonesense-ink" data-testid="nhanes-n-high">
              {nHigh} of {result.models.length}
            </div>
            <div className="text-xs text-stonesense-ink/50">
              at the {point === 'accuracy' ? 'accuracy-focused' : 'balanced'} cut-offs · other point:{' '}
              {result.n_high_by_point?.[point === 'accuracy' ? 'balanced' : 'accuracy'] ?? '—'} of {result.models.length}
            </div>
          </div>
          <div className="rounded-md border border-stonesense-line bg-stonesense-paper p-3.5">
            <div className="text-xs text-stonesense-ink/50">Agreement</div>
            <div className={`mt-1 text-xl font-semibold ${agree ? 'text-stonesense-teal' : 'text-stonesense-amber'}`}>
              {agree ? 'Models agree' : 'Models disagree'}
            </div>
          </div>
        </div>

        {result.warnings.length > 0 ? (
          <ul className="mt-3 list-disc pl-5 text-xs text-stonesense-amber">
            {result.warnings.map((w) => (
              <li key={w}>{w}</li>
            ))}
          </ul>
        ) : null}

        <div className="mt-5" role="radiogroup" aria-label="Operating point" data-testid="nhanes-operating-point-toggle">
          <div className="flex flex-wrap gap-2">
            {OPERATING_POINT_OPTIONS.map((o) => (
              <button
                key={o.key}
                type="button"
                role="radio"
                aria-checked={point === o.key}
                onClick={() => setPoint(o.key)}
                title={o.hint}
                className={`rounded-md border px-3 py-1.5 text-xs font-medium ${
                  point === o.key
                    ? 'border-stonesense-teal bg-stonesense-teal text-white'
                    : 'border-stonesense-line bg-white text-stonesense-ink hover:bg-black/[0.03]'
                }`}
              >
                {o.label}
              </button>
            ))}
          </div>
          {point === 'accuracy' ? (
            <p
              className="mt-2 rounded-md border border-stonesense-amber/40 bg-stonesense-amber/10 px-3 py-2 text-xs text-stonesense-ink/80"
              role="note"
              data-testid="nhanes-accuracy-caution"
            >
              <strong>Caution:</strong> the accuracy-focused point flags fewer patients as High risk, so it misses more true cases. Its accuracy is above
              70% mainly because most people have no stone (majority baseline {pct(baseline)}); on held-out data it finds only{' '}
              {sensRange.length ? `${pct(Math.min(...sensRange))} to ${pct(Math.max(...sensRange))}` : '—'} of true cases (sensitivity).
            </p>
          ) : (
            <p className="mt-2 text-xs text-stonesense-ink/60">
              Balanced point: finds more true cases (higher sensitivity) at the price of lower accuracy and many more High-risk flags.
            </p>
          )}
        </div>

        <div className="mt-4 overflow-x-auto">
          <table className="w-full min-w-[720px] text-sm" data-testid="nhanes-comparison-table">
            <caption className="caption-top pb-2 text-left text-xs text-stonesense-ink/60">
              <strong>Left: this patient.</strong> Right (shaded): model performance on held-out {testCycle} test data
              (n = {result.models[0]?.test_metrics.n.toLocaleString()}), <em>not this patient&apos;s accuracy</em>. Hover a value for its 95% CI.
            </caption>
            <thead>
              <tr className="border-b border-stonesense-line text-left text-xs text-stonesense-ink/60">
                <th className="px-3 py-2 font-medium">Model</th>
                <th className="px-3 py-2 text-right font-medium">Risk %</th>
                <th className="px-3 py-2 font-medium">Risk level</th>
                <th className="bg-stonesense-paper px-3 py-2 text-right font-medium">ROC-AUC</th>
                <th className="bg-stonesense-paper px-3 py-2 text-right font-medium">
                  Accuracy
                  <span className="block text-[10px] font-normal text-stonesense-ink/50">majority baseline {pct(baseline)}</span>
                </th>
                <th className="bg-stonesense-paper px-3 py-2 text-right font-semibold text-stonesense-ink">
                  Sensitivity
                  <span className="block text-[10px] font-normal text-stonesense-ink/50">true cases found</span>
                </th>
                <th className="bg-stonesense-paper px-3 py-2 text-right font-medium">Specificity</th>
                <th className="bg-stonesense-paper px-3 py-2 text-right font-medium">Balanced acc.</th>
              </tr>
            </thead>
            <tbody>
              {result.models.map((m) => {
                const op = opOf(m, point)?.test ?? undefined;
                const isSel = m.model_id === selected?.model_id;
                return (
                  <tr
                    key={m.model_id}
                    onClick={() => setSelectedId(m.model_id)}
                    className={`cursor-pointer border-b border-stonesense-line/70 ${isSel ? 'bg-stonesense-teal/5' : 'hover:bg-black/[0.02]'}`}
                    aria-selected={isSel}
                  >
                    <td className="px-3 py-2 font-medium text-stonesense-ink">
                      {m.model_name}
                      {isSel ? <span className="ml-2 text-[10px] uppercase text-stonesense-teal">selected</span> : null}
                    </td>
                    <td className="px-3 py-2 text-right tabular-nums font-semibold">{pct(m.probability)}</td>
                    <td className="px-3 py-2">
                      <span
                        className={`rounded-full px-2.5 py-0.5 text-xs font-semibold ${
                          riskAt(m) === 'High' ? 'bg-stonesense-amber/15 text-stonesense-amber' : 'bg-stonesense-teal/15 text-stonesense-teal'
                        }`}
                      >
                        {riskAt(m)}
                      </span>
                      <span className="ml-2 text-[10px] text-stonesense-ink/40">≥ {pct(thresholdAt(m))}</span>
                    </td>
                    <td className="bg-stonesense-paper px-3 py-2 text-right tabular-nums" title={ciText(m.test_metrics.roc_auc, m.test_metrics.roc_auc_ci95)}>
                      {m.test_metrics.roc_auc.toFixed(3)}
                    </td>
                    <MetricCell value={op?.accuracy} ci={op?.accuracy_ci95} strong />
                    <MetricCell value={op?.sensitivity} ci={op?.sensitivity_ci95} strong />
                    <MetricCell value={op?.specificity} ci={op?.specificity_ci95} />
                    <MetricCell value={op?.balanced_accuracy} ci={op?.balanced_accuracy_ci95} />
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <p className="mt-3 text-xs text-stonesense-ink/60">{result.metrics_note}</p>
        <p className="mt-1 text-xs text-stonesense-ink/60">
          About {pct(baseline)} of the {testCycle} participants report no stone, so &quot;no stone for everyone&quot; already scores {pct(baseline)} accuracy;
          read accuracy together with sensitivity. Metrics are at the selected operating point&apos;s cut-off, which each model chose on the 2015–2016
          validation cycle only ({point === 'accuracy' ? 'highest sensitivity with validation accuracy ≥ 72%' : 'maximum balanced accuracy'}).
        </p>

        {selected ? (
          <details className="mt-3 text-xs text-stonesense-ink/70">
            <summary className="cursor-pointer font-medium text-stonesense-ink">Both operating points, per model (held-out test data)</summary>
            <ul className="mt-2 space-y-2">
              {result.models.map((m) => (
                <li key={m.model_id}>
                  <strong>{m.model_name}</strong>
                  {(['accuracy', 'balanced'] as NhanesOperatingPointKey[]).map((k) => (
                    <div key={k} className="ml-3">
                      <span className="text-stonesense-ink/50">{opOf(m, k)?.label}:</span>{' '}
                      <OperatingPointLine m={opOf(m, k)?.test ?? null} baseline={m.test_metrics.majority_baseline_accuracy} />
                    </div>
                  ))}
                </li>
              ))}
            </ul>
          </details>
        ) : null}
      </div>

      {selected ? (
        <div className="rounded-lg border border-stonesense-line bg-white p-6 shadow-sm">
          <p className="text-xs font-semibold uppercase tracking-wider text-stonesense-teal">Explanation · {selected.model_name}</p>
          <h3 className="mt-1 font-serif text-base text-stonesense-ink">Top contributing factors for this patient</h3>
          <ol className="mt-3 space-y-2">
            {selected.top_factors.map((f) => (
              <li key={f.feature} className="text-sm">
                <div className="flex items-baseline justify-between gap-3">
                  <span className="text-stonesense-ink">
                    {f.label}
                    {f.value_label ? (
                      <span className="ml-2 text-xs text-stonesense-ink/50">{f.value_label}</span>
                    ) : f.value !== null ? (
                      <span className="ml-2 text-xs text-stonesense-ink/50">{Number(f.value.toFixed(2))}</span>
                    ) : null}
                  </span>
                  <span className={`text-xs tabular-nums ${f.direction === 'increases' ? 'text-stonesense-amber' : 'text-stonesense-teal'}`}>
                    {f.direction === 'increases' ? 'raises' : f.direction === 'decreases' ? 'lowers' : 'no effect on'} risk ({f.contribution >= 0 ? '+' : ''}
                    {f.contribution.toFixed(2)} log-odds)
                  </span>
                </div>
                <div className="mt-1 h-1.5 w-full overflow-hidden rounded-full bg-stonesense-line">
                  <div
                    className={`h-full rounded-full ${f.contribution >= 0 ? 'bg-stonesense-amber' : 'bg-stonesense-teal'}`}
                    style={{ width: `${Math.max(3, (Math.abs(f.contribution) / maxContribution) * 100)}%` }}
                  />
                </div>
              </li>
            ))}
          </ol>
          <p className="mt-3 text-xs text-stonesense-ink/60">{selected.explanation_summary} Select another row above to see that model&apos;s factors.</p>
        </div>
      ) : null}

      <p className="rounded-lg border border-stonesense-amber/40 bg-stonesense-amber/10 p-4 text-xs text-stonesense-ink/80" role="note">
        {result.disclaimer}
      </p>
    </section>
  );
}
