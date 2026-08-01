export function LoadingSpinner({ label = 'Loading...' }: { label?: string }) {
  return (
    <div className="flex items-center gap-3 rounded-2xl border border-cyan-500/30 bg-slate-900/70 px-4 py-3 text-sm text-cyan-100">
      <span className="h-4 w-4 animate-spin rounded-full border-2 border-cyan-400 border-t-transparent" />
      <span>{label}</span>
    </div>
  );
}
