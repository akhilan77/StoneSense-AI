// Drop at: frontend/src/context/HospitalContext.tsx
//
// Holds the currently-selected hospital (dropdown, no auth yet per current
// project stage). Every hospital-scoped API call reads hospitalId from here.
// When real auth lands, this Provider's `setHospitalId` call site becomes
// "read hospital_id out of the login response" — no consumer changes.

import { useState, useEffect, ReactNode } from "react";
import { Hospital } from "../types/dashboard";
import { fetchHospitals } from "../services/hospitalApi";
import { HospitalContext } from "./hospitalContextDef";

export function HospitalProvider({ children }: { children: ReactNode }) {
  const [hospitals, setHospitals] = useState<Hospital[]>([]);
  const [hospitalId, setHospitalIdState] = useState<number | null>(() => {
    const stored = localStorage.getItem("stonesense_hospital_id");
    return stored ? Number(stored) : null;
  });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchHospitals()
      .then((list) => {
        setHospitals(list);
        if (!hospitalId && list.length > 0) {
          setHospitalIdState(list[0].id);
        }
      })
      .finally(() => setLoading(false));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const setHospitalId = (id: number) => {
    setHospitalIdState(id);
    localStorage.setItem("stonesense_hospital_id", String(id));
  };

  return (
    <HospitalContext.Provider value={{ hospitals, hospitalId, setHospitalId, loading }}>
      {children}
    </HospitalContext.Provider>
  );
}

