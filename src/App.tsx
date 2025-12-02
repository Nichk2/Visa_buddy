import React from 'react';
import { BrowserRouter as Router, Routes, Route, useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import Test from './components/ChatInterface'

function App() {
  return (
    <Router>
      <AnimatePresence>
      <div className="min-h-screen from-blue-50 to-indigo-100">
        <Routes>
          <Route path="/" element={<Test />} />
        </Routes>
      </div>
      </AnimatePresence>
    </Router>
  );
}

export default App;