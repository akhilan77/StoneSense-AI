import axios, { AxiosError, InternalAxiosRequestConfig } from 'axios';
import { clearStoredSession, getStoredRefreshToken, getStoredToken } from './authApi';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? '';

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor: Automatically attach Bearer token to all protected outgoing requests
apiClient.interceptors.request.use(
  (config: InternalAxiosRequestConfig) => {
    const token = getStoredToken();
    if (token && !config.headers.Authorization) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Response interceptor: Handle 401 Unauthorized, refresh token, or redirect to /login
apiClient.interceptors.response.use(
  (response) => response,
  async (error: AxiosError) => {
    const originalRequest = error.config as InternalAxiosRequestConfig & { _retry?: boolean };

    if (error.response?.status === 401 && originalRequest) {
      const url = originalRequest.url ?? '';
      
      // Do not attempt refresh on auth endpoints to prevent endless loops
      if (url.includes('/auth/login') || url.includes('/auth/refresh')) {
        return Promise.reject(error);
      }

      if (!originalRequest._retry) {
        originalRequest._retry = true;
        const refreshToken = getStoredRefreshToken();

        if (refreshToken) {
          try {
            const refreshResponse = await axios.post<{ access_token: string }>(
              `${API_BASE_URL}/api/v1/auth/refresh`,
              { refresh_token: refreshToken },
              { headers: { 'Content-Type': 'application/json' } }
            );

            const newAccessToken = refreshResponse.data.access_token;
            localStorage.setItem('stonesense_access_token', newAccessToken);
            originalRequest.headers.Authorization = `Bearer ${newAccessToken}`;

            return apiClient(originalRequest);
          } catch (refreshErr) {
            clearStoredSession();
            if (window.location.pathname !== '/login') {
              window.location.href = '/login';
            }
            return Promise.reject(refreshErr);
          }
        } else {
          clearStoredSession();
          if (window.location.pathname !== '/login') {
            window.location.href = '/login';
          }
        }
      }
    }

    return Promise.reject(error);
  }
);

export default apiClient;
