import { useState, useEffect } from "react";
import { Routes, Route } from "react-router-dom";
import Dashboard from './dashboard'
import AccessNotice from './AccessNotice'

function App() {
  const [showNotice, setShowNotice] = useState(false);

  useEffect(() => {
    const checkWidth = () => {
      setShowNotice(window.innerWidth < 768);
    };
    checkWidth();
    window.addEventListener('resize', checkWidth);
    return () => window.removeEventListener('resize', checkWidth);
  }, []);

  return (
    <>
      {showNotice && <AccessNotice />}
      <Routes>
        {/* Public Routes */}
        <Route path="/" element={<Dashboard />} />
      </Routes>
    </>
  );
}

export default App;