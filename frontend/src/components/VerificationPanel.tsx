import { useState } from 'react';
import { Download, Upload, CheckCircle, AlertTriangle } from 'lucide-react';
import axios from 'axios';

interface VerificationPanelProps {
    smiles: string;
    aiDensity: number;
}

export const VerificationPanel = ({ smiles, aiDensity }: VerificationPanelProps) => {
    const [lammpsDensity, setLammpsDensity] = useState<number | null>(null);
    const [errorMargin, setErrorMargin] = useState<number | null>(null);
    const [loading, setLoading] = useState(false);

    const handleDownload = async () => {
        try {
            const res = await axios.post('http://localhost:5000/api/lammps/generate', { smiles }, {
                responseType: 'blob' // Important: Expect binary data
            });
            const url = window.URL.createObjectURL(new Blob([res.data]));
            const a = document.createElement('a');
            a.href = url;
            a.download = 'lammps_pkg.zip'; // Changed extension
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
        } catch (e) {
            console.error("Download failed", e);
        }
    };

    const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) return;

        const text = await file.text();
        setLoading(true);
        try {
            const res = await axios.post('http://localhost:5000/api/lammps/parse', { content: text });
            const realDensity = res.data.density;
            setLammpsDensity(realDensity);

            const error = Math.abs((realDensity - aiDensity) / realDensity) * 100;
            setErrorMargin(error);
        } catch (e) {
            console.error("Parse failed", e);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="bg-glass border border-white/10 rounded-xl p-6 space-y-6">
            <h3 className="text-gray-400 font-mono text-sm uppercase flex items-center gap-2">
                <CheckCircle className="w-4 h-4 text-green-400" />
                Ground Truth Verification
            </h3>

            <div className="grid grid-cols-2 gap-4">
                <button
                    onClick={handleDownload}
                    className="flex justify-center items-center gap-2 bg-white/5 hover:bg-white/10 p-4 rounded-lg border border-white/5 transition-colors"
                >
                    <Download className="w-5 h-5 text-blue-400" />
                    <span className="text-sm">Download Input</span>
                </button>

                <label className="flex justify-center items-center gap-2 bg-white/5 hover:bg-white/10 p-4 rounded-lg border border-white/5 transition-colors cursor-pointer">
                    <Upload className="w-5 h-5 text-purple-400" />
                    <span className="text-sm">{loading ? 'Analyzing...' : 'Upload Log'}</span>
                    <input type="file" className="hidden" onChange={handleFileUpload} />
                </label>
            </div>

            {lammpsDensity && (
                <div className="bg-black/40 p-4 rounded-lg border border-white/5 space-y-2">
                    <div className="flex justify-between items-center">
                        <span className="text-gray-400">LAMMPS Density</span>
                        <span className="font-mono text-white">{lammpsDensity.toFixed(3)} g/cm³</span>
                    </div>
                    <div className="flex justify-between items-center">
                        <span className="text-gray-400">AI Prediction</span>
                        <span className="font-mono text-white">{aiDensity.toFixed(3)} g/cm³</span>
                    </div>
                    <div className="h-px bg-white/10 my-2" />
                    <div className="flex justify-between items-center">
                        <span className="text-gray-400">Error Margin</span>
                        <span className={`font-bold ${errorMargin! < 5 ? 'text-green-400' : 'text-red-400'}`}>
                            {errorMargin!.toFixed(2)}%
                        </span>
                    </div>
                    {errorMargin! < 5 && (
                        <div className="mt-2 flex items-center justify-center gap-2 text-green-400 bg-green-400/10 p-2 rounded">
                            <CheckCircle className="w-4 h-4" />
                            <span className="text-xs font-bold uppercase tracking-widest">Verified Accurate</span>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};
