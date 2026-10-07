export type UserRole = 'hospital_user' | 'developer' | 'admin';

export interface AuthUser {
  id: number;
  email: string;
  role: UserRole;
  hospital_id?: number | null;
  hospital_code?: string | null;
  is_active: boolean;
  created_at: string;
}

export interface LoginResponse {
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in_seconds: number;
  user: AuthUser;
}

export interface RefreshResponse {
  access_token: string;
  token_type: string;
  expires_in_seconds: number;
}
