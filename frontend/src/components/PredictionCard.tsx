interface PredictionCardProps {
  title: string;
  data: Array<{
    label: string;
    value: string | number | boolean;
  }>;
}

export function PredictionCard({ title, data }: PredictionCardProps) {
  return (
    <section className="rounded-3xl border border-slate-800 bg-slate-900/80 p-5 shadow-xl shadow-slate-950/40">
      <div className="mb-4 flex items-center justify-between">
        <h3 className="text-lg font-semibold text-white">{title}</h3>
      </div>
      <div className="grid gap-3 sm:grid-cols-2">
        {data.map((item) => (
          <div key={item.label} className="rounded-2xl bg-slate-800/70 p-3 text-sm">
            <div className="text-slate-400">{item.label}</div>
            <div className="mt-1 text-base font-semibold text-white">
              {typeof item.value === 'boolean' ? (item.value ? 'Yes' : 'No') : item.value}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
