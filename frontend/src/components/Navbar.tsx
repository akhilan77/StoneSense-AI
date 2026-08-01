import { useEffect, useState } from 'react';
import { NavLink } from 'react-router-dom';
import { health } from '../services/api';

const navItems = [
  { to: '/', label: 'Home' },
  { to: '/risk-prediction', label: 'Risk Prediction' },
  { to: '/stone-detection', label: 'Stone Detection' },
  { to: '/assessment', label: 'Assessment' },
];

export function Navbar() {
  const [backendStatus, setBackendStatus] = useState<'online' | 'offline'>('offline');

  useEffect(() => {
    void health()
      .then(() => setBackendStatus('online'))
      .catch(() => setBackendStatus('offline'));
  }, []);

  return (
    <header className="sticky top-0 z-30 border-b border-slate-800 bg-slate-950/90 backdrop-blur">
      <div className="mx-auto flex max-w-7xl flex-col gap-3 px-4 py-4 sm:px-6 lg:flex-row lg:items-center lg:justify-between lg:px-8">
        <div>
          <div className="text-lg font-semibold text-white">StoneSense AI</div>
          <div className="text-sm text-slate-400">Explainable kidney stone detection and risk prediction</div>
        </div>

        <nav className="flex flex-wrap items-center gap-2">
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) =>
                `rounded-full px-3 py-2 text-sm transition ${
                  isActive ? 'bg-cyan-400 text-slate-950' : 'bg-slate-900 text-slate-200 hover:bg-slate-800'
                }`
              }
            >
              {item.label}
            </NavLink>
          ))}
          <span
            className={`rounded-full px-3 py-2 text-sm font-medium ${
              backendStatus === 'online' ? 'bg-emerald-500/20 text-emerald-300' : 'bg-rose-500/20 text-rose-300'
            }`}
          >
            {backendStatus === 'online' ? 'Backend Online' : 'Backend Offline'}
          </span>
        </nav>
      </div>
    </header>
  );
}
