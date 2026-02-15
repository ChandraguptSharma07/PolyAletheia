/**
 * PolyAletheia Client Application.
 * 
 * Main entry point for the React frontend.
 * Features:
 * - "The Void" Aesthetic: Minimalist, dark-mode UI with Framer Motion transitions.
 * - 3D Molecule Visualization: Integrated React Three Fiber viewer.
 * - Real-time Inference: Connects to Express + Python backend.
 * - LAMMPS Verification: Ground truth comparison workflow.
 */

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Search, Atom, Zap, Activity } from 'lucide-react';
import axios from 'axios';
import { MoleculeViewer } from './components/MoleculeViewer';
import { VerificationPanel } from './components/VerificationPanel';
import { TokenHeatmap } from './components/TokenHeatmap';

// --- Configuration ---
const API_BASE_URL = 'http://localhost:5000/api';

// --- Types ---
interface PredictionProperties {
  Tg: number;       // Glass Transition Temp (°C)
  Tc: number;       // Crystallization Temp (°C)
  Density: number;  // Density (g/cm³)
  FFV: number;      // Fractional Free Volume
  Rg: number;       // Radius of Gyration (Å)
}

interface AtomData {
  idx: number;
  symbol: string;
  pos: [number, number, number];
  color: string;
}

interface BondData {
  start: number;
  end: number;
  order: number;
}

interface MoleculeStructure {
  atoms: AtomData[];
  bonds: BondData[];
  weights?: any;
}

function App() {
  // State
  // We keep it simple with local state for now. 
  // If this grows, we might need a context or Redux, but valid for a prototype.
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionProperties | null>(null);
  const [structure, setStructure] = useState<MoleculeStructure | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeFeature, setActiveFeature] = useState<string | null>(null);

  // Handlers
  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query) return;

    setLoading(true);
    setError(null);
    setResult(null);
    setStructure(null);
    setActiveFeature(null);

    try {
      // Connect to Backend Inference Service
      const response = await axios.post(`${API_BASE_URL}/predict`, { smiles: query });

      if (response.data.success && response.data.properties) {
        setResult(response.data.properties);
        setStructure(response.data.structure);
      } else if (response.data.error) {
        throw new Error(response.data.error);
      } else {
        throw new Error("Invalid response format from server");
      }
    } catch (err) {
      console.error("Prediction failed:", err);
      setError("Failed to analyze molecule. Please ensure the backend is running.");
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setResult(null);
    setStructure(null);
    setError(null);
    setQuery('');
    setActiveFeature(null);
  };

  return (
    <div className="min-h-screen bg-void text-white font-sans flex flex-col items-center justify-center p-4 selection:bg-blue-500 selection:text-white overflow-hidden relative">

      {/* Background Glow Effect - subtle ambiance to match the "Void" theme */}
      <div className="absolute top-[-20%] left-[-20%] w-[140%] h-[140%] bg-[radial-gradient(circle_at_center,_var(--tw-gradient-stops))] from-blue-900/20 via-black to-black z-0 pointer-events-none" />

      {/* Main Content Area with Smooth Transitions */}
      <AnimatePresence mode='wait'>

        {/* State 1: Search / Hero */}
        {!result && (
          <motion.div
            key="hero"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="z-10 w-full max-w-2xl text-center space-y-8"
          >
            <motion.h1
              className="text-6xl md:text-8xl font-bold tracking-tighter bg-clip-text text-transparent bg-gradient-to-b from-white to-white/40"
              initial={{ scale: 0.9 }}
              animate={{ scale: 1 }}
              transition={{ duration: 0.8, ease: "circOut" }}
            >
              PolyAletheia
            </motion.h1>

            <p className="text-xl text-gray-400 font-light">
              The Architecture of Matter.
            </p>

            <form onSubmit={handleSearch} className="relative group">
              {/* Glowing Border Effect */}
              <div className="absolute -inset-0.5 bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg blur opacity-30 group-hover:opacity-75 transition duration-1000 group-hover:duration-200" />

              <div className="relative flex items-center bg-black border border-white/10 rounded-lg p-2">
                <Search className="ml-3 text-gray-400 w-5 h-5" />
                <input
                  type="text"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="Enter SMILES or describe a polymer..."
                  className="w-full bg-transparent text-white placeholder-gray-600 px-4 py-3 focus:outline-none text-lg font-mono"
                />
                <button
                  type="submit"
                  disabled={loading}
                  className="px-6 py-2 bg-white text-black font-semibold rounded-md hover:bg-gray-200 transition-colors disabled:opacity-50"
                >
                  {loading ? ' thinking...' : 'Analyze'}
                </button>
              </div>
              {error && <p className="text-red-400 text-sm mt-2 font-mono">{error}</p>}
            </form>
          </motion.div>
        )}

        {/* State 2: Dashboard / Results */}
        {result && (
          <motion.div
            key="results"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="z-10 w-full max-w-6xl grid grid-cols-1 md:grid-cols-3 gap-6"
          >
            {/* 3D Molecule Viewer */}
            {/* Left Column: 3D Model + Token Heatmap */}
            <div className="md:col-span-2 flex flex-col gap-6">
              {/* 3D Molecule Viewer */}
              <div className="h-[500px] bg-glass rounded-2xl border border-white/10 relative overflow-hidden group flex flex-col">

                {/* Feature Selection Dropdown */}
                <div className="absolute top-4 left-4 z-20">
                  <div className="relative group">
                    <select
                      value={activeFeature || ""}
                      onChange={(e) => setActiveFeature(e.target.value || null)}
                      className="appearance-none bg-black/60 backdrop-blur-md border border-white/20 text-white pl-4 pr-10 py-2 rounded-lg font-mono text-sm focus:outline-none focus:border-blue-500 hover:bg-black/80 transition-colors cursor-pointer"
                    >
                      <option value="" className="bg-black text-gray-300">Standard View</option>
                      <option value="Tg" className="bg-black font-semibold">Highlight: Glass Transition</option>
                      <option value="Tc" className="bg-black font-semibold">Highlight: Melting Point</option>
                      <option value="Density" className="bg-black font-semibold">Highlight: Density</option>
                      <option value="FFV" className="bg-black font-semibold">Highlight: Free Volume</option>
                      <option value="Rg" className="bg-black font-semibold">Highlight: Gyration Radius</option>
                    </select>
                    {/* Custom Arrow */}
                    <div className="absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none text-gray-400">
                      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="6 9 12 15 18 9"></polyline></svg>
                    </div>
                  </div>
                </div>

                <div className="absolute inset-0 bg-gradient-to-tr from-blue-500/10 to-purple-500/10 opacity-30 pointer-events-none" />
                <MoleculeViewer structure={structure} activeFeature={activeFeature} />
              </div>

              {/* Token Heatmap Visualization */}
              <TokenHeatmap weights={structure?.weights} activeFeature={activeFeature} />
            </div>

            {/* Property Cards */}
            <div className="space-y-4">
              <PropertyCard
                icon={<Activity className="w-5 h-5 text-blue-400" />}
                label="Glass Transition"
                value={result.Tg.toFixed(1)}
                unit="°C"
                delay={0.1}
              />
              <PropertyCard
                icon={<Zap className="w-5 h-5 text-yellow-400" />}
                label="Melting Point"
                value={result.Tc.toFixed(1)}
                unit="°C"
                delay={0.2}
              />
              <PropertyCard
                icon={<Atom className="w-5 h-5 text-purple-400" />}
                label="Density"
                value={result.Density.toFixed(3)}
                unit="g/cm³"
                delay={0.3}
              />
              <PropertyCard
                icon={<Activity className="w-5 h-5 text-green-400" />}
                label="Free Volume"
                value={result.FFV.toFixed(3)}
                unit="fraction"
                delay={0.4}
              />
              <PropertyCard
                icon={<Activity className="w-5 h-5 text-orange-400" />}
                label="Gyration Radius"
                value={result.Rg.toFixed(2)}
                unit="Å"
                delay={0.5}
              />

              <button
                onClick={handleReset}
                className="w-full py-4 mt-8 bg-white/5 border border-white/10 text-white rounded-lg hover:bg-white/10 transition-colors font-mono text-sm"
              >
                ESC / NEW SEARCH
              </button>

              <div className="mt-8">
                <VerificationPanel smiles={query} aiDensity={result.Density} />
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// Sub-component for cleaner JSX
const PropertyCard = ({ icon, label, value, unit, delay }: { icon: React.ReactNode, label: string, value: string, unit: string, delay: number }) => (
  <motion.div
    initial={{ x: 20, opacity: 0 }}
    animate={{ x: 0, opacity: 1 }}
    transition={{ delay }}
    className="bg-glass p-6 rounded-xl border border-white/10"
  >
    <div className="flex items-center space-x-3 mb-2">
      {icon}
      <h3 className="text-gray-400 text-sm font-mono uppercase">{label}</h3>
    </div>
    <p className="text-4xl font-bold">{value} <span className="text-lg text-gray-600">{unit}</span></p>
  </motion.div>
);

export default App;
