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
    <section className="rounded-lg border border-stonesense-line bg-white p-6 shadow-sm">
      <div className="mb-4 flex items-center justify-between">
        <h3 className="font-serif text-lg text-stonesense-ink">{title}</h3>
        {numericVal !== null ? (
          <span className={`rounded-full px-3 py-1 text-xs font-semibold ${
            numericVal > 75 ? 'bg-stonesense-teal/15 text-stonesense-teal' : 'bg-stonesense-amber/15 text-stonesense-amber'
          }`}>
            {numericVal}% Match
          </span>
        ) : null}
      </div>

      <div className="grid gap-3 sm:grid-cols-2">
        {data.map((item) => (
          <div key={item.label} className="rounded-md border border-stonesense-line bg-stonesense-paper p-3.5">
            <div className="text-xs text-stonesense-ink/50">{item.label}</div>
            <div className="mt-1 text-base font-semibold text-stonesense-ink">
              {typeof item.value === 'boolean' ? (item.value ? 'Yes' : 'No') : item.value}
            </div>
          </div>
        ))}
      </div>

      {numericVal !== null ? (
        <div className="mt-4 h-1.5 w-full overflow-hidden rounded-full bg-stonesense-line">
          <div 
            className={`h-full rounded-full transition-all duration-500 ${
              numericVal > 75 ? 'bg-stonesense-teal' : 'bg-stonesense-amber'
            }`}
            style={{ width: `${numericVal}%` }}
          />
        </div>
      ) : null}
    </section>
  );
}
