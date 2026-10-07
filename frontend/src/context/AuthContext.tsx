import React, { createContext, useContext, useEffect, useState } from 'react';
import axios from 'axios';
import { AuthUser, UserRole } from '../types/auth';
import {
  clearStoredSession,
  fetchCurrentProfile,
  getStoredRefreshToken,
  getStoredToken,
  getStoredUser,
  loginApi,
  refreshApi,
  setStoredSession,
} from '../services/authApi';

interface AuthContextValue {
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

const AuthContext = createContext<AuthContextValue | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(() => getStoredUser());
  const [token, setToken] = useState<string | null>(() => getStoredToken());
  const [loading, setLoading] = useState(true);

  // Setup Axios global request interceptor to attach JWT token
  useEffect(() => {
    const requestInterceptor = axios.interceptors.request.use(
      (config) => {
        const activeToken = getStoredToken();
        if (activeToken && !config.headers.Authorization) {
          config.headers.Authorization = `Bearer ${activeToken}`;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );

    const responseInterceptor = axios.interceptors.response.use(
      (response) => response,
      async (error) => {
        if (error.response?.status === 401) {
          const originalRequest = error.config;
          const refreshToken = getStoredRefreshToken();
          if (refreshToken && !originalRequest._retry && !originalRequest.url?.includes('/auth/')) {
            originalRequest._retry = true;
            try {
              const refreshData = await refreshApi(refreshToken);
              localStorage.setItem('stonesense_access_token', refreshData.access_token);
              setToken(refreshData.access_token);
              originalRequest.headers.Authorization = `Bearer ${refreshData.access_token}`;
              return axios(originalRequest);
            } catch {
              clearStoredSession();
              setUser(null);
              setToken(null);
              window.location.href = '/login';
            }
          } else if (!originalRequest.url?.includes('/auth/login')) {
            clearStoredSession();
            setUser(null);
            setToken(null);
          }
        }
        return Promise.reject(error);
      }
    );

    return () => {
      axios.interceptors.request.eject(requestInterceptor);
      axios.interceptors.response.eject(responseInterceptor);
    };
  }, []);

  // Validate session on load
  useEffect(() => {
    const existingToken = getStoredToken();
    if (existingToken) {
      fetchCurrentProfile()
        .then((profile) => {
          setUser(profile);
          if (profile.hospital_id) {
            localStorage.setItem('stonesense_hospital_id', String(profile.hospital_id));
          }
        })
        .catch(() => {
          clearStoredSession();
          setUser(null);
          setToken(null);
        })
        .finally(() => setLoading(false));
    } else {
      setLoading(false);
    }
  }, []);

  const login = async (email: string, password: string) => {
    const res = await loginApi(email, password);
    setStoredSession(res.access_token, res.refresh_token, res.user);
    setToken(res.access_token);
    setUser(res.user);
  };

  const logout = () => {
    clearStoredSession();
    setUser(null);
    setToken(null);
    window.location.href = '/login';
  };

  const refresh = async () => {
    const storedRefresh = getStoredRefreshToken();
    if (!storedRefresh) throw new Error('No refresh token available');
    const res = await refreshApi(storedRefresh);
    localStorage.setItem('stonesense_access_token', res.access_token);
    setToken(res.access_token);
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        token,
        role: user?.role ?? null,
        hospitalId: user?.hospital_id ?? null,
        isAuthenticated: Boolean(token && user),
        loading,
        login,
        logout,
        refresh,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}
