// Drop at: frontend/src/components/layout/Sidebar.tsx
import { NavLink } from "react-router-dom";

interface SidebarProps {
  role: "hospital" | "developer";
}

const hospitalLinks = [
  { to: "/hospital-dashboard", label: "Overview" },
  { to: "/risk-prediction", label: "Risk prediction" },
  { to: "/stone-detection", label: "Stone detection" },
];

const developerLinks = [
  { to: "/developer-dashboard", label: "Model performance" },
  { to: "/developer-dashboard/versions", label: "Versions & deployment" },
  { to: "/developer-dashboard/hospitals", label: "Hospital update logs" },
  { to: "/developer-dashboard/monitoring", label: "System monitoring" },
  { to: "/developer-dashboard/drift", label: "Drift analysis" },
];

export default function Sidebar({ role }: SidebarProps) {
  const links = role === "hospital" ? hospitalLinks : developerLinks;
  const accent = role === "hospital" ? "stonesense-teal" : "stonesense-indigo";

  return (
    <aside className="flex h-full w-60 shrink-0 flex-col justify-between bg-stonesense-ink text-stonesense-paper">
      <div>
        <div className="px-5 pt-6 pb-5">
          <p className="font-serif text-lg leading-tight">StoneSense</p>
          <p className="text-xs text-stonesense-paper/50 mt-0.5">
            {role === "hospital" ? "Hospital console" : "Developer console"}
          </p>
        </div>
        <nav className="mt-1 flex flex-col gap-0.5 px-3">
          {links.map((link) => (
            <NavLink
              key={link.to}
              to={link.to}
              end
              className={({ isActive }) =>
                `rounded-md px-3 py-2 text-sm transition-colors ${
                  isActive
                    ? `bg-${accent}/15 text-white font-medium`
                    : "text-stonesense-paper/70 hover:bg-white/5 hover:text-white"
                }`
              }
            >
              {link.label}
            </NavLink>
          ))}
        </nav>
      </div>
      <div className="px-5 py-4 text-[11px] text-stonesense-paper/40 border-t border-white/10">
        Evidence-based support only — not a clinical diagnosis.
      </div>
    </aside>
  );
}
