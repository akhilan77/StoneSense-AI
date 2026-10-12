import { createContext } from 'react';
import { AuthUser, UserRole } from '../types/auth';

export interface AuthContextValue {
  user: AuthUser | null;
  token: string | null;
  role: UserRole | null;
  hospitalId: number | null;
  isAuthenticated: boolean;
  loading: boolean;
  login: (email: string, password: string) => Promise<void>;
  logout: () => void;
  refresh: () => Promise<void>;
}

export const AuthContext = createContext<AuthContextValue | undefined>(undefined);
