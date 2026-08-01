import { Navigate, Route, Routes } from 'react-router-dom';
import { MainLayout } from './layouts/MainLayout';
import { HomePage } from './pages/HomePage';
import { RiskPredictionPage } from './pages/RiskPredictionPage';
import { StoneDetectionPage } from './pages/StoneDetectionPage';
import { AssessmentPage } from './pages/AssessmentPage';
import { NotFoundPage } from './pages/NotFoundPage';

export function App() {
  return (
    <MainLayout>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/risk-prediction" element={<RiskPredictionPage />} />
        <Route path="/stone-detection" element={<StoneDetectionPage />} />
        <Route path="/assessment" element={<AssessmentPage />} />
        <Route path="/404" element={<NotFoundPage />} />
        <Route path="*" element={<Navigate to="/404" replace />} />
      </Routes>
    </MainLayout>
  );
}
