import React from 'react';
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App.jsx';
import './index.css';
import { LevaPanel } from "leva";

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
    <LevaPanel />
  </StrictMode>,
);
