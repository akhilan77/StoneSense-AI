import { Navigate, Route, Routes } from 'react-router-dom';
import { ProtectedRoute } from './components/ProtectedRoute';
import { MainLayout } from './layouts/MainLayout';
import DeveloperDashboard from './pages/DeveloperDashboard';
import { HomePage } from './pages/HomePage';
import HospitalDashboard from './pages/HospitalDashboard';
import HospitalFederatedLearning from './pages/HospitalFederatedLearning';
import HospitalPrivateDataset from './pages/HospitalPrivateDataset';
import { LandingPage } from './pages/LandingPage';
import { LoginPage } from './pages/LoginPage';
import { NotFoundPage } from './pages/NotFoundPage';
import { PrivacyPolicy } from './pages/PrivacyPolicy';
import { RiskPredictionPage } from './pages/RiskPredictionPage';
import { StoneDetectionPage } from './pages/StoneDetectionPage';

export function App() {
  return (
    <Routes>
      {/* Public landing and auth routes */}
      <Route path="/" element={<LandingPage />} />
      <Route path="/privacy" element={<PrivacyPolicy />} />
      <Route path="/login" element={<LoginPage />} />

      {/* Hospital Console Routes (Scoped to hospital_user, developer, admin) */}
      <Route
        path="/hospital-dashboard"
        element={
          <ProtectedRoute allowedRoles={['hospital_user', 'developer', 'admin']}>
            <HospitalDashboard />
          </ProtectedRoute>
        }
      />
      <Route
        path="/hospital/federated-learning"
        element={
          <ProtectedRoute allowedRoles={['hospital_user', 'developer', 'admin']}>
            <HospitalFederatedLearning />
          </ProtectedRoute>
        }
      />
      <Route
        path="/hospital/private-dataset"
        element={
          <ProtectedRoute allowedRoles={['hospital_user', 'developer', 'admin']}>
            <HospitalPrivateDataset />
          </ProtectedRoute>
        }
      />
      <Route
        path="/risk-prediction/:patientId"
        element={
          <ProtectedRoute allowedRoles={['hospital_user', 'developer', 'admin']}>
            <RiskPredictionPage />
          </ProtectedRoute>
        }
      />
      <Route
        path="/stone-detection/:patientId"
        element={
          <ProtectedRoute allowedRoles={['hospital_user', 'developer', 'admin']}>
            <StoneDetectionPage />
          </ProtectedRoute>
        }
      />

      {/* Developer Console Routes (Guarded strictly for developer and admin) */}
      <Route
        path="/developer-dashboard"
        element={
          <ProtectedRoute allowedRoles={['developer', 'admin']}>
            <DeveloperDashboard initialTab="overview" />
          </ProtectedRoute>
        }
      />
      <Route
        path="/developer-dashboard/federated"
        element={
          <ProtectedRoute allowedRoles={['developer', 'admin']}>
            <DeveloperDashboard initialTab="federated" />
          </ProtectedRoute>
        }
      />
      <Route
        path="/developer-dashboard/versions"
        element={
          <ProtectedRoute allowedRoles={['developer', 'admin']}>
            <DeveloperDashboard initialTab="versions" />
          </ProtectedRoute>
        }
      />
      <Route
        path="/developer-dashboard/access"
        element={
          <ProtectedRoute allowedRoles={['developer', 'admin']}>
            <DeveloperDashboard initialTab="access" />
          </ProtectedRoute>
        }
      />
      <Route
        path="/developer-dashboard/health"
        element={
          <ProtectedRoute allowedRoles={['developer', 'admin']}>
            <DeveloperDashboard initialTab="health" />
          </ProtectedRoute>
        }
      />

      {/* Public Home / Marketing Layout */}
      <Route
        path="/home"
        element={
          <MainLayout>
            <HomePage />
          </MainLayout>
        }
      />

      {/* Fallback */}
      <Route path="/404" element={<NotFoundPage />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
