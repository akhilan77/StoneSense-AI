// Drop at: frontend/src/pages/Login.tsx
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useHospital } from "../context/HospitalContext";

export default function Login() {
  const navigate = useNavigate();
  const { hospitals, hospitalId, setHospitalId } = useHospital();
  const [mode, setMode] = useState<"select" | "hospital" | "developer">("select");

  const enterHospital = () => {
    if (hospitalId == null && hospitals.length > 0) {
      setHospitalId(hospitals[0].id);
    }
    navigate("/hospital-dashboard");
  };

  const enterDeveloper = () => {
    navigate("/developer-dashboard");
  };

  return (
    <div className="flex h-screen items-center justify-center bg-stonesense-paper text-stonesense-ink">
      <div className="w-full max-w-md">
        <div className="mb-8 text-center">
          <h1 className="font-serif text-3xl">StoneSense</h1>
          <p className="text-sm text-stonesense-ink/60 mt-1">
            Sign in to continue
          </p>
        </div>

        {mode === "select" && (
          <div className="flex flex-col gap-3">
            <button
              onClick={() => setMode("hospital")}
              className="rounded-lg border border-stonesense-line bg-white p-5 text-left hover:border-stonesense-teal transition-colors"
            >
              <p className="font-serif text-lg">Hospital login</p>
              <p className="text-sm text-stonesense-ink/60 mt-1">
                Submit patient cases and review risk / imaging results.
              </p>
            </button>

            <button
              onClick={() => setMode("developer")}
              className="rounded-lg border border-stonesense-line bg-white p-5 text-left hover:border-stonesense-indigo transition-colors"
            >
              <p className="font-serif text-lg">Developer login</p>
              <p className="text-sm text-stonesense-ink/60 mt-1">
                Monitor model performance, deployments, and system health.
              </p>
            </button>
          </div>
        )}

        {mode === "hospital" && (
          <div className="rounded-lg border border-stonesense-line bg-white p-6">
            <p className="font-serif text-lg mb-1">Hospital login</p>
            <p className="text-sm text-stonesense-ink/60 mb-4">
              Select your hospital to continue.
            </p>
            <select
              value={hospitalId ?? ""}
              onChange={(e) => setHospitalId(Number(e.target.value))}
              className="w-full rounded-md border border-stonesense-line px-3 py-2 text-sm mb-4"
            >
              {hospitals.map((h) => (
                <option key={h.id} value={h.id}>
                  {h.name}
                </option>
              ))}
            </select>
            <button
              onClick={enterHospital}
              className="w-full rounded-md bg-stonesense-teal px-4 py-2 text-sm text-white"
            >
              Continue
            </button>
            <button
              onClick={() => setMode("select")}
              className="w-full mt-2 text-xs text-stonesense-ink/50 underline"
            >
              Back
            </button>
          </div>
        )}

        {mode === "developer" && (
          <div className="rounded-lg border border-stonesense-line bg-white p-6">
            <p className="font-serif text-lg mb-1">Developer login</p>
            <p className="text-sm text-stonesense-ink/60 mb-4">
              No credentials required yet — access is open during this phase.
            </p>
            <button
              onClick={enterDeveloper}
              className="w-full rounded-md bg-stonesense-indigo px-4 py-2 text-sm text-white"
            >
              Continue
            </button>
            <button
              onClick={() => setMode("select")}
              className="w-full mt-2 text-xs text-stonesense-ink/50 underline"
            >
              Back
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
