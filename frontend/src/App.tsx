import { Navigate, Route, Routes } from 'react-router-dom';
import { MainLayout } from './layouts/MainLayout';
import { HomePage } from './pages/HomePage';
import { NotFoundPage } from './pages/NotFoundPage';
import { RiskPredictionPage } from './pages/RiskPredictionPage';
import { StoneDetectionPage } from './pages/StoneDetectionPage';
import HospitalDashboard from './pages/HospitalDashboard';
import DeveloperDashboard from './pages/DeveloperDashboard';
import { LandingPage } from './pages/LandingPage';
import { PrivacyPolicy } from './pages/PrivacyPolicy';
import HospitalFederatedLearning from './pages/HospitalFederatedLearning';

export function App() {
  return (
    <Routes>
      {/* Root landing page */}
      <Route path="/" element={<LandingPage />} />
      <Route path="/privacy" element={<PrivacyPolicy />} />
      <Route path="/login" element={<Navigate to="/" replace />} />

      {/* Hospital Console Routes (All use persistent left vertical sidebar) */}
      <Route path="/hospital-dashboard" element={<HospitalDashboard />} />
      <Route path="/hospital/federated-learning" element={<HospitalFederatedLearning />} />
      <Route path="/risk-prediction/:patientId" element={<RiskPredictionPage />} />
      <Route path="/stone-detection/:patientId" element={<StoneDetectionPage />} />

      {/* Developer Console Routes (All use persistent left vertical sidebar) */}
      <Route path="/developer-dashboard" element={<DeveloperDashboard initialTab="overview" />} />
      <Route path="/developer-dashboard/federated" element={<DeveloperDashboard initialTab="federated" />} />
      <Route path="/developer-dashboard/versions" element={<DeveloperDashboard initialTab="versions" />} />
      <Route path="/developer-dashboard/access" element={<DeveloperDashboard initialTab="access" />} />
      <Route path="/developer-dashboard/health" element={<DeveloperDashboard initialTab="health" />} />

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
