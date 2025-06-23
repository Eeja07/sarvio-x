import { Routes, Route } from "react-router-dom";
import Dashboard from './dashboard'

function App() {
  return (
    <Routes>
      {/* Public Routes */}
      <Route path="/" element={<Dashboard />} />
    </Routes>
  );
}

export default App;