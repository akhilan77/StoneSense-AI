import axios from 'axios';
import { AuthUser, LoginResponse, RefreshResponse } from '../types/auth';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? '';

export const authClient = axios.create({
  baseURL: `${API_BASE_URL}/api/v1/auth`,
  headers: {
    'Content-Type': 'application/json',
  },
});

export function getStoredToken(): string | null {
  return localStorage.getItem('stonesense_access_token');
}

export function getStoredRefreshToken(): string | null {
  return localStorage.getItem('stonesense_refresh_token');
}

export function getStoredUser(): AuthUser | null {
  const raw = localStorage.getItem('stonesense_user');
  if (!raw) return null;
  try {
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

export function setStoredSession(token: string, refreshToken: string, user: AuthUser): void {
  localStorage.setItem('stonesense_access_token', token);
  localStorage.setItem('stonesense_refresh_token', refreshToken);
  localStorage.setItem('stonesense_user', JSON.stringify(user));
  if (user.hospital_id) {
    localStorage.setItem('stonesense_hospital_id', String(user.hospital_id));
  }
}

export function clearStoredSession(): void {
  localStorage.removeItem('stonesense_access_token');
  localStorage.removeItem('stonesense_refresh_token');
  localStorage.removeItem('stonesense_user');
}

export async function loginApi(email: string, password: string): Promise<LoginResponse> {
  const { data } = await authClient.post<LoginResponse>('/login', { email, password });
  return data;
}

export async function refreshApi(refreshToken: string): Promise<RefreshResponse> {
  const { data } = await authClient.post<RefreshResponse>('/refresh', { refresh_token: refreshToken });
  return data;
}

export async function fetchCurrentProfile(): Promise<AuthUser> {
  const token = getStoredToken();
  const { data } = await authClient.get<AuthUser>('/me', {
    headers: token ? { Authorization: `Bearer ${token}` } : {},
  });
  return data;
}
