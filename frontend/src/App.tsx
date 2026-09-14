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

export function App() {
  return (
    <Routes>
      {/* Root landing page */}
      <Route path="/" element={<LandingPage />} />
      <Route path="/privacy" element={<PrivacyPolicy />} />
      <Route path="/login" element={<Navigate to="/" replace />} />

      {/* Hospital Console Routes (All use persistent left vertical sidebar) */}
      <Route path="/hospital-dashboard" element={<HospitalDashboard />} />
      <Route path="/risk-prediction" element={<RiskPredictionPage />} />
      <Route path="/stone-detection" element={<StoneDetectionPage />} />

      {/* Developer Console Routes (All use persistent left vertical sidebar) */}
      <Route path="/developer-dashboard" element={<DeveloperDashboard initialTab="overview" />} />
      <Route path="/developer-dashboard/versions" element={<DeveloperDashboard initialTab="versions" />} />
      <Route path="/developer-dashboard/hospitals" element={<DeveloperDashboard initialTab="hospitals" />} />
      <Route path="/developer-dashboard/monitoring" element={<DeveloperDashboard initialTab="monitoring" />} />
      <Route path="/developer-dashboard/drift" element={<DeveloperDashboard initialTab="drift" />} />
      <Route path="/developer-dashboard/enrolled-hospitals" element={<DeveloperDashboard initialTab="access" />} />
      <Route path="/developer-dashboard/access" element={<DeveloperDashboard initialTab="access" />} />

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
