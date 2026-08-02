import { Link } from 'react-router-dom';

export function NotFoundPage() {
  return (
    <div className="flex min-h-[60vh] flex-col items-center justify-center gap-4 rounded-3xl border border-slate-800 bg-slate-900/70 p-8 text-center">
      <div className="text-5xl">404</div>
      <h1 className="text-2xl font-semibold text-white">Page not found</h1>
      <p className="max-w-lg text-slate-300">
        The requested route is unavailable in the current StoneSense AI front end.
      </p>
      <Link
        to="/"
        className="rounded-full bg-cyan-500 px-5 py-2 font-medium text-slate-950 transition hover:bg-cyan-400"
      >
        Return home
      </Link>
    </div>
  );
}
