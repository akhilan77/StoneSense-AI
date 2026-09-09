// Drop at: frontend/src/components/layout/AppLayout.tsx
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
    <div className="flex h-screen bg-[#F3F6F1] text-[#101B16] overflow-hidden">
      <Sidebar role={role} />
      <div className="flex-1 overflow-y-auto">
        <header className="topbar flex items-center justify-between border-b border-[#DDE3DC] px-8 py-5 bg-[#F3F6F1]">
          <div>
            <h1 className="font-serif text-2xl font-medium text-[#101B16]">{title}</h1>
            {subtitle && <p className="text-xs text-[#101B16]/60 mt-1">{subtitle}</p>}
          </div>

          <div className="flex items-center gap-3">
            {role === "hospital" && (
              <select
                value={hospitalId ?? ""}
                onChange={(e) => setHospitalId(Number(e.target.value))}
                className="rounded-md border border-[#DDE3DC] bg-white px-3 py-1.5 text-xs text-[#101B16]/80 cursor-pointer shadow-xs"
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
              className="rounded-md border border-[#DDE3DC] bg-white px-3 py-1.5 text-xs text-[#101B16]/75 hover:bg-black/5 cursor-pointer shadow-xs"
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
