export interface LammpsResult {
    density: number;
    error_margin?: number;
}

export const parseLog = (logContent: string): LammpsResult => {
    // Simple regex to find density (This is a simplified example)
    // Looking for lines like: "Loop time of 12.34 on 8 procs..."
    // Or extracting density from thermo output

    // Mock logic: Look for "Density" keyword or just assume standard thermo output
    // Real parser would handle multi-column thermo data

    const lines = logContent.split('\n');
    let densities: number[] = [];
    let densityIndex = -1;

    // Scan the log for Thermo data
    for (let i = 0; i < lines.length; i++) {
        const line = lines[i].trim();

        // Find Header
        if (line.startsWith('Step') && (line.includes('Density') || line.includes('density'))) {
            const headers = line.split(/\s+/);
            densityIndex = headers.findIndex(h => h.toLowerCase() === 'density');
            continue;
        }

        // Parse Data Rows
        if (densityIndex !== -1 && /^\d+/.test(line)) {
            const cols = line.split(/\s+/);
            // Ensure verify we have enough columns and valid number
            if (cols.length > densityIndex) {
                const val = parseFloat(cols[densityIndex]);
                if (!isNaN(val)) {
                    densities.push(val);
                }
            }
        }

        // Stop if we hit loop summary or errors
        if (line.startsWith('Loop time') || line.startsWith('ERROR')) {
            densityIndex = -1;
        }
    }

    // Calculate Average of last 50% of steps (Production Phase)
    if (densities.length > 0) {
        const productionData = densities.slice(Math.floor(densities.length / 2));
        const avgDensity = productionData.reduce((a, b) => a + b, 0) / productionData.length;

        return { density: avgDensity };
    }

    // Fallback if no density found (e.g. old logs without 'thermo_style custom')
    // We try to find "Density" keyword in standard output if present
    // But for now, returning 0 or error is safer than mock.
    return { density: 0, error_margin: 100 };
};
