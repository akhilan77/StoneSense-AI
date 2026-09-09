import { useEffect, useState } from 'react';
import { NavLink } from 'react-router-dom';
import { health } from '../services/api';

const navItems = [
  { to: '/login', label: 'Login' },
  { to: '/hospital-dashboard', label: 'Hospital Console' },
  { to: '/developer-dashboard', label: 'Developer Console' },
  { to: '/risk-prediction', label: 'Risk Prediction' },
  { to: '/stone-detection', label: 'Stone Detection' },
];

export function Navbar() {
  const [backendStatus, setBackendStatus] = useState<'online' | 'offline'>('offline');

  useEffect(() => {
    void health()
      .then(() => setBackendStatus('online'))
      .catch(() => setBackendStatus('offline'));
  }, []);

  return (
    <header className="sticky top-0 z-30 border-b border-stonesense-line bg-white/90 backdrop-blur">
      <div className="mx-auto flex max-w-7xl flex-col gap-3 px-4 py-4 sm:px-6 lg:flex-row lg:items-center lg:justify-between lg:px-8">
        <div>
          <div className="font-serif text-xl font-semibold text-stonesense-ink">StoneSense AI</div>
          <div className="text-sm text-stonesense-ink/60">
            Explainable kidney stone detection and risk prediction
          </div>
        </div>

        <nav className="flex flex-wrap items-center gap-2">
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) =>
                `rounded-md px-3 py-1.5 text-sm transition ${
                  isActive
                    ? 'bg-stonesense-teal text-white font-medium'
                    : 'bg-stonesense-ink/5 text-stonesense-ink/80 hover:bg-stonesense-ink/10'
                }`
              }
            >
              {item.label}
            </NavLink>
          ))}
          <span
            className={`rounded-md px-3 py-1.5 text-sm font-medium ${
              backendStatus === 'online'
                ? 'bg-stonesense-teal/15 text-stonesense-teal'
                : 'bg-rose-500/15 text-rose-700'
            }`}
          >
            {backendStatus === 'online' ? 'Backend Online' : 'Backend Offline'}
          </span>
        </nav>
      </div>
    </header>
  );
}
