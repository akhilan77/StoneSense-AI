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

  return (
    <aside className="sidebar flex h-full w-60 shrink-0 flex-col justify-between bg-[#101B16] text-[#F3F6F1]">
      <div>
        <div className="px-5 pt-6 pb-4">
          <p className="font-serif text-lg leading-tight text-white font-medium">StoneSense</p>
          <p className="text-xs text-[#F3F6F1]/50 mt-0.5">
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
                    ? `${
                        role === "hospital"
                          ? "active role-hospital bg-[#1F6F5C]/25 text-white font-medium"
                          : "active role-developer bg-[#3B3F8C]/30 text-white font-medium"
                      }`
                    : "text-[#F3F6F1]/70 hover:bg-white/10 hover:text-white"
                }`
              }
            >
              {link.label}
            </NavLink>
          ))}
        </nav>
      </div>
      <div className="px-5 py-4 text-[11px] text-[#F3F6F1]/35 border-t border-white/10">
        Evidence-based support only — not a clinical diagnosis.
      </div>
    </aside>
  );
}
