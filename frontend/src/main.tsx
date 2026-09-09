import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import { App } from './App';
import { HospitalProvider } from './context/HospitalContext';
import './index.css';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <BrowserRouter>
      <HospitalProvider>
        <App />
      </HospitalProvider>
    </BrowserRouter>
  </React.StrictMode>
);