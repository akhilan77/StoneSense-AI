// Drop at: frontend/src/components/layout/AppLayout.tsx
// (replaces the previous version — the demo "Switch to X view" button is gone,
// replaced with a Log out link that returns to /login)
import { ReactNode } from "react";
import { useNavigate } from "react-router-dom";
import Sidebar from "./Sidebar";
import { useHospital } from "../../context/HospitalContext";

interface AppLayoutProps {
  role: "hospital" | "developer";
  title: string;
  subtitle?: string;
  children: ReactNode;
}

export default function AppLayout({ role, title, subtitle, children }: AppLayoutProps) {
  const navigate = useNavigate();
  const { hospitals, hospitalId, setHospitalId } = useHospital();

  return (
    <div className="flex h-screen bg-stonesense-paper text-stonesense-ink">
      <Sidebar role={role} />
      <div className="flex-1 overflow-y-auto">
        <header className="flex items-center justify-between border-b border-stonesense-line px-8 py-5">
          <div>
            <h1 className="font-serif text-2xl">{title}</h1>
            {subtitle && <p className="text-sm text-stonesense-ink/60 mt-0.5">{subtitle}</p>}
          </div>

          <div className="flex items-center gap-3">
            {role === "hospital" && (
              <select
                value={hospitalId ?? ""}
                onChange={(e) => setHospitalId(Number(e.target.value))}
                className="rounded-md border border-stonesense-line bg-white px-3 py-1.5 text-sm"
              >
                {hospitals.map((h) => (
                  <option key={h.id} value={h.id}>
                    {h.name}
                  </option>
                ))}
              </select>
            )}

            <button
              onClick={() => navigate("/login")}
              className="rounded-md border border-stonesense-line px-3 py-1.5 text-sm text-stonesense-ink/70 hover:bg-stonesense-ink/5"
            >
              Log out
            </button>
          </div>
        </header>

        <main className="px-8 py-6">{children}</main>
      </div>
    </div>
  );
}
