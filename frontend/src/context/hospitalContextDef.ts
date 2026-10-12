import { createContext } from 'react';
import { Hospital } from '../types/dashboard';

export interface HospitalContextValue {
  hospitals: Hospital[];
  hospitalId: number | null;
  setHospitalId: (id: number) => void;
  loading: boolean;
}

export const HospitalContext = createContext<HospitalContextValue | undefined>(undefined);
