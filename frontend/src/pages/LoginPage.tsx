import React, { useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

export function LoginPage() {
  const { login } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const isDevMode = import.meta.env.MODE !== 'production';

  const demoAccounts = [
    {
      role: 'Admin',
      name: 'System Administrator',
      email: 'admin@stonesense.ai',
      password: 'StoneSenseAdmin!2026',
      badge: 'bg-rose-500/10 text-rose-400 border-rose-500/20',
    },
    {
      role: 'Developer',
      name: 'ML Research Lead',
      email: 'dev@stonesense.ai',
      password: 'StoneSenseDev!2026',
      badge: 'bg-indigo-500/10 text-indigo-400 border-indigo-500/20',
    },
    {
      role: 'Hospital 1',
      name: 'Apollo Renal Care (HOSP-001)',
      email: 'hospital1@stonesense.ai',
      password: 'Hospital1!2026',
      badge: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20',
    },
    {
      role: 'Hospital 2',
      name: 'Fortis Nephrology (HOSP-002)',
      email: 'hospital2@stonesense.ai',
      password: 'Hospital2!2026',
      badge: 'bg-cyan-500/10 text-cyan-400 border-cyan-500/20',
    },
    {
      role: 'Hospital 3',
      name: 'AIIMS Urology (HOSP-003)',
      email: 'hospital3@stonesense.ai',
      password: 'Hospital3!2026',
      badge: 'bg-amber-500/10 text-amber-400 border-amber-500/20',
    },
  ];

  const handleSubmit = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!email || !password) {
      setError('Please enter both email and password.');
      return;
    }

    setError(null);
    setLoading(true);

    try {
      await login(email.trim(), password);
      const from = (location.state as { from?: { pathname?: string } })?.from?.pathname;
      if (from && from !== '/login') {
        navigate(from, { replace: true });
      } else if (email.includes('hospital')) {
        navigate('/hospital-dashboard', { replace: true });
      } else {
        navigate('/developer-dashboard', { replace: true });
      }
    } catch (err: any) {
      const msg = err.response?.data?.detail || 'Authentication failed. Please check your credentials.';
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  const handleQuickLogin = (demoEmail: string, demoPass: string) => {
    setEmail(demoEmail);
    setPassword(demoPass);
    setError(null);
    setLoading(true);

    login(demoEmail, demoPass)
      .then(() => {
        const from = (location.state as { from?: { pathname?: string } })?.from?.pathname;
        if (from && from !== '/login') {
          navigate(from, { replace: true });
        } else if (demoEmail.includes('hospital')) {
          navigate('/hospital-dashboard', { replace: true });
        } else {
          navigate('/developer-dashboard', { replace: true });
        }
      })
      .catch((err: any) => {
        const msg = err.response?.data?.detail || 'Quick login failed.';
        setError(msg);
      })
      .finally(() => setLoading(false));
  };

  return (
    <div className="flex min-h-screen bg-slate-950 text-slate-100 selection:bg-emerald-500 selection:text-white">
      {/* Background radial gradient */}
      <div className="fixed inset-0 pointer-events-none bg-[radial-gradient(ellipse_80%_80%_at_50%_-20%,rgba(16,185,129,0.12),rgba(255,255,255,0))]" />

      <div className="relative mx-auto flex w-full max-w-6xl flex-col items-center justify-center p-6 lg:flex-row lg:gap-16">
        {/* Left column: Branding & Overview */}
        <div className="w-full max-w-md pb-8 lg:pb-0">
          <div className="flex items-center gap-3">
            <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-gradient-to-tr from-emerald-600 to-teal-400 font-bold text-white shadow-lg shadow-emerald-500/20">
              <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
              </svg>
            </div>
            <div>
              <h1 className="text-2xl font-bold tracking-tight text-white">StoneSense-AI</h1>
              <p className="text-xs font-medium uppercase tracking-wider text-emerald-400">Federated Clinical Network</p>
            </div>
          </div>

          <h2 className="mt-8 text-3xl font-bold tracking-tight text-slate-100 sm:text-4xl">
            Secure Clinical <br />
            <span className="bg-gradient-to-r from-emerald-400 to-teal-300 bg-clip-text text-transparent">
              Access Control & Auth
            </span>
          </h2>
          <p className="mt-4 text-sm leading-relaxed text-slate-400">
            Isolated tenant routing for hospital federated learning, verifiable model governance, and confidential patient records.
          </p>

          <div className="mt-8 space-y-3">
            <div className="flex items-center gap-3 text-xs text-slate-300">
              <span className="flex h-5 w-5 items-center justify-center rounded-full bg-emerald-500/10 text-emerald-400">✓</span>
              Cryptographic JWT validation with short-lived tokens
            </div>
            <div className="flex items-center gap-3 text-xs text-slate-300">
              <span className="flex h-5 w-5 items-center justify-center rounded-full bg-emerald-500/10 text-emerald-400">✓</span>
              Strict tenant isolation & zero cross-hospital leakage
            </div>
            <div className="flex items-center gap-3 text-xs text-slate-300">
              <span className="flex h-5 w-5 items-center justify-center rounded-full bg-emerald-500/10 text-emerald-400">✓</span>
              Immutable audit logging for clinical reads and deployment gates
            </div>
          </div>
        </div>

        {/* Right column: Login form & Demo shortcuts */}
        <div className="w-full max-w-md rounded-2xl border border-slate-800 bg-slate-900/80 p-8 shadow-2xl backdrop-blur-xl">
          <h3 className="text-xl font-semibold text-white">Sign In</h3>
          <p className="mt-1 text-xs text-slate-400">Enter your clinical or developer credentials to continue</p>

          {error && (
            <div className="mt-4 rounded-lg border border-rose-500/30 bg-rose-500/10 p-3 text-xs text-rose-300">
              {error}
            </div>
          )}

          <form onSubmit={handleSubmit} className="mt-6 space-y-4">
            <div>
              <label className="block text-xs font-medium text-slate-300">Email Address</label>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="name@hospital.org"
                required
                className="mt-1.5 w-full rounded-lg border border-slate-700 bg-slate-950 px-3.5 py-2.5 text-sm text-white placeholder-slate-500 transition focus:border-emerald-500 focus:outline-none focus:ring-1 focus:ring-emerald-500"
              />
            </div>

            <div>
              <label className="block text-xs font-medium text-slate-300">Password</label>
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••••••"
                required
                className="mt-1.5 w-full rounded-lg border border-slate-700 bg-slate-950 px-3.5 py-2.5 text-sm text-white placeholder-slate-500 transition focus:border-emerald-500 focus:outline-none focus:ring-1 focus:ring-emerald-500"
              />
            </div>

            <button
              type="submit"
              disabled={loading}
              className="mt-2 flex w-full items-center justify-center rounded-lg bg-emerald-600 px-4 py-2.5 text-sm font-semibold text-white shadow-md shadow-emerald-900/30 transition hover:bg-emerald-500 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:ring-offset-2 focus:ring-offset-slate-900 disabled:opacity-50"
            >
              {loading ? (
                <div className="flex items-center gap-2">
                  <div className="h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
                  <span>Signing in...</span>
                </div>
              ) : (
                'Sign In to StoneSense'
              )}
            </button>
          </form>

          {/* Quick Login Section (Non-production mode only) */}
          {isDevMode && (
            <div className="mt-8 border-t border-slate-800 pt-6">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">Demo Role Quick-Fill</span>
                <span className="rounded bg-slate-800 px-1.5 py-0.5 text-[10px] font-mono text-emerald-400">Dev Mode</span>
              </div>
              <p className="mt-1 text-[11px] text-slate-500">Click any role to authenticate instantly with seeded demo accounts:</p>

              <div className="mt-3 grid gap-2">
                {demoAccounts.map((acc) => (
                  <button
                    key={acc.email}
                    type="button"
                    onClick={() => handleQuickLogin(acc.email, acc.password)}
                    disabled={loading}
                    className="flex items-center justify-between rounded-lg border border-slate-800/80 bg-slate-950/60 p-2.5 text-left transition hover:border-slate-700 hover:bg-slate-800/50"
                  >
                    <div>
                      <div className="text-xs font-medium text-slate-200">{acc.name}</div>
                      <div className="text-[11px] text-slate-500">{acc.email}</div>
                    </div>
                    <span className={`rounded border px-2 py-0.5 text-[10px] font-semibold ${acc.badge}`}>
                      {acc.role}
                    </span>
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
