/**
 * TokenHeatmap.tsx
 * 
 * Visualizes the AI's "attention" as a spectral analysis graph.
 * 
 * We use a bar-chart visualization (like a mass spec or NMR readout) because
 * it's more scientifically intuitive than a simple heat-mapped text string.
 * This allows researchers to immediately spot the "peaks" (critical substructures)
 * that drive the physical property prediction.
 */

import { motion } from 'framer-motion';

interface TokenData {
    tokens: string[];
    scores: number[];
}

interface TokenHeatmapProps {
    weights: any;
    activeFeature: string | null;
}

export const TokenHeatmap = ({ weights, activeFeature }: TokenHeatmapProps) => {
    // We need data to render. If weights are missing entirely, we can't show anything.
    if (!weights) return null;

    // Use active feature data OR fallback to first available property just to get the tokens
    // The tokens themselves are the same for all properties.
    const tokenKey = activeFeature ? `${activeFeature}_tokens` : Object.keys(weights).find(k => k.endsWith('_tokens'));
    const data: TokenData | undefined = tokenKey ? weights[tokenKey] : undefined;

    if (!data || !data.tokens) return null;

    // If no feature is selected, maxScore is 1 (to avoid division by zero), and scores are effectively ignored
    const maxScore = activeFeature && data.scores ? Math.max(...data.scores, 0.00001) : 1;

    return (
        <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="w-full bg-glass border border-white/10 rounded-xl p-6 mt-6 backdrop-blur-md"
        >
            <h3 className="text-gray-400 text-sm font-mono mb-4 uppercase tracking-wider">
                Token Attribution: <span className="text-white font-bold">{activeFeature || "Standard View"}</span>
            </h3>

            {/* Spectrum / Equalizer View */}
            <div className="relative w-full overflow-x-auto pb-8 custom-scrollbar">
                <div className="flex items-end justify-start min-w-full h-40 px-4 space-x-[2px] pt-10"> {/* Added pt-10 for tooltip space */}
                    {data.tokens.map((token, index) => {
                        // Filter out control tokens
                        if (['<s>', '</s>', '<pad>'].includes(token)) return null;

                        // Calculate logic:
                        // If no feature selected: score = 0, intensity = 0
                        const score = (activeFeature && data.scores) ? data.scores[index] : 0;
                        const intensity = activeFeature ? Math.max(0, score / maxScore) : 0;

                        // Height calculation
                        // If inactive: Flat line (4px)
                        // If active: Scaled height
                        const isHighImpact = intensity > 0.4;
                        const heightPercent = activeFeature ? Math.max(10, intensity * 100) : 4;

                        // Color styling
                        // If active: Gradient from White (top) to Pink (bottom) for that "energy" feel.
                        // If inactive: subtle ghost bars to keep layout stable.
                        const barColor = activeFeature
                            ? (isHighImpact ? `linear-gradient(to top, #ff00ff, #ffffff)` : `rgba(255, 255, 255, 0.15)`)
                            : `rgba(255, 255, 255, 0.05)`;

                        const cleanToken = token.replace('Ġ', '');

                        return (
                            <div key={index} className="flex flex-col items-center group relative w-8 flex-shrink-0 h-full justify-end">
                                {/* Tooltip - Only show impact if active */}
                                {activeFeature && (
                                    <div className="absolute bottom-[calc(100%+8px)] left-1/2 -translate-x-1/2 hidden group-hover:block z-30 bg-black border border-white/20 p-2 rounded text-xs whitespace-nowrap pointer-events-none shadow-xl">
                                        <div className="font-bold text-[#ff00ff] text-center mb-1">{cleanToken}</div>
                                        <div className="text-gray-300">Impact: <span className="text-white font-mono">{(intensity * 100).toFixed(0)}%</span></div>
                                    </div>
                                )}

                                {/* The Bar */}
                                <motion.div
                                    initial={{ height: "4px" }}
                                    animate={{ height: activeFeature ? `${heightPercent}%` : "4px" }}
                                    transition={{ duration: 0.5, delay: index * 0.01 }}
                                    className={`w-full rounded-t-sm transition-all duration-300 ${(activeFeature && isHighImpact) ? 'shadow-[0_0_15px_#ff00ff]' : 'hover:bg-white/10'}`}
                                    style={{
                                        background: barColor,
                                    }}
                                />

                                {/* The Token Label */}
                                <div className={`absolute top-full mt-2 left-1/2 -translate-x-1/2 text-[10px] font-mono whitespace-nowrap ${(activeFeature && isHighImpact) ? 'text-white font-bold' : 'text-gray-600'}`}>
                                    {cleanToken}
                                </div>
                            </div>
                        );
                    })}
                </div>
            </div>

            <div className="mt-8 flex items-center justify-between text-xs text-gray-500 font-mono border-t border-white/10 pt-2">
                <div>SPECTRAL ANALYSIS // {activeFeature || "READY"}</div>
                <div className="flex items-center gap-4 opacity-50">
                    <span className="flex items-center gap-1"><div className="w-2 h-2 bg-white/10 rounded-full" /> Baseline</span>
                    <span className="flex items-center gap-1"><div className="w-2 h-2 bg-[#ff00ff] shadow-[0_0_5px_#ff00ff] rounded-full" /> Driver</span>
                </div>
            </div>
        </motion.div>
    );
};
