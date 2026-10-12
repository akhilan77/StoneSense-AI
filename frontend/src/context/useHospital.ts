import { useContext } from 'react';
import { HospitalContext } from './hospitalContextDef';

export function useHospital() {
  const ctx = useContext(HospitalContext);
  if (!ctx) throw new Error('useHospital must be used within a HospitalProvider');
  return ctx;
}
