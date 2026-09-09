import { Navigate, Route, Routes } from 'react-router-dom';
import { MainLayout } from './layouts/MainLayout';
import { HomePage } from './pages/HomePage';
import { NotFoundPage } from './pages/NotFoundPage';
import { RiskPredictionPage } from './pages/RiskPredictionPage';
import { StoneDetectionPage } from './pages/StoneDetectionPage';
import HospitalDashboard from './pages/HospitalDashboard';
import DeveloperDashboard from './pages/DeveloperDashboard';
import Login from './pages/Login';

export function App() {
  return (
    <Routes>
      <Route path="/login" element={<Login />} />
      <Route path="/hospital-dashboard" element={<HospitalDashboard />} />
      <Route path="/developer-dashboard" element={<DeveloperDashboard />} />
      <Route
        path="*"
        element={
          <MainLayout>
            <Routes>
              <Route path="/" element={<Navigate to="/login" replace />} />
              <Route path="/home" element={<HomePage />} />
              <Route path="/risk-prediction" element={<RiskPredictionPage />} />
              <Route path="/stone-detection" element={<StoneDetectionPage />} />
              <Route path="/404" element={<NotFoundPage />} />
              <Route path="*" element={<Navigate to="/404" replace />} />
            </Routes>
          </MainLayout>
        }
      />
    </Routes>
  );
}
