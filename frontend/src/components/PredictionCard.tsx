interface PredictionCardProps {
  title: string;
  data: Array<{
    label: string;
    value: string | number | boolean;
  }>;
}

export function PredictionCard({ title, data }: PredictionCardProps) {
  // Find confidence or probability to render custom visual indicator
  const confidenceItem = data.find(d => d.label.toLowerCase().includes('confidence') || d.label.toLowerCase().includes('probability'));
  const numericVal = confidenceItem ? parseFloat(String(confidenceItem.value).replace('%', '')) : null;

  return (
    <section className="group rounded-3xl border border-slate-800 bg-slate-900/60 p-6 shadow-xl shadow-slate-950/40 transition-all duration-300 hover:border-cyan-500/50 hover:bg-slate-900/80">
      <div className="mb-4 flex items-center justify-between">
        <h3 className="text-lg font-bold text-white tracking-wide">{title}</h3>
        {numericVal !== null ? (
          <span className={`rounded-full px-3 py-1 text-xs font-bold tracking-wider ${
            numericVal > 75 ? 'bg-emerald-500/15 text-emerald-400' : 'bg-amber-500/15 text-amber-400'
          }`}>
            {numericVal}% Match
          </span>
        ) : null}
      </div>

      <div className="grid gap-3 sm:grid-cols-2">
        {data.map((item) => (
          <div key={item.label} className="rounded-2xl bg-slate-950/65 border border-slate-900 p-4 transition-colors hover:border-slate-800/80">
            <div className="text-xs font-semibold uppercase tracking-wider text-slate-500">{item.label}</div>
            <div className="mt-1.5 text-base font-semibold text-white">
              {typeof item.value === 'boolean' ? (item.value ? 'Yes' : 'No') : item.value}
            </div>
          </div>
        ))}
      </div>

      {numericVal !== null ? (
        <div className="mt-5 w-full bg-slate-950 rounded-full h-1.5 overflow-hidden">
          <div 
            className={`h-full rounded-full transition-all duration-500 ${
              numericVal > 75 ? 'bg-gradient-to-r from-cyan-400 to-emerald-400' : 'bg-gradient-to-r from-amber-400 to-orange-400'
            }`}
            style={{ width: `${numericVal}%` }}
          />
        </div>
      ) : null}
    </section>
  );
}
