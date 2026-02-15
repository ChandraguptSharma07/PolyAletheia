/**
 * PolyAletheia Backend Server.
 * 
 * Orchestrates the interaction between the React Frontend and the Python Inference Engine.
 * Handles API requests for property prediction, LAMMPS input generation, and log parsing.
 */

import express, { Request, Response, NextFunction } from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import { PythonShell, Options as PythonShellOptions } from 'python-shell';
import path from 'path';
import { generateInput } from './lammps/generator';
import { parseLog } from './lammps/parser';

// Load environment variables
dotenv.config();

const app = express();
const PORT = process.env.PORT || 5000;
const PYTHON_PATH = process.env.PYTHON_PATH || 'python'; // Allow override for different environments

// Middleware
app.use(cors());
app.use(express.json());

// --- Configuration ---
const INFERENCE_SCRIPT_PATH = path.resolve(__dirname, '../../src/inference');
const INFERENCE_SCRIPT_NAME = 'predict_api.py';

// --- Types ---
interface InferenceRequest {
    smiles: string;
}

interface LammpsGenerateRequest {
    smiles: string;
    temp?: number;
    pressure?: number;
}

interface LammpsParseRequest {
    content: string;
}

// --- Routes ---

/**
 * Health Check Endpoint
 */
app.get('/', (req: Request, res: Response) => {
    res.json({
        status: 'active',
        system: 'PolyAletheia Backend',
        version: '2.0.0'
    });
});

/**
 * Inference Endpoint
 * Spawns a Python process to run the inference script.
 */
app.post('/api/predict', async (req: Request<{}, {}, InferenceRequest>, res: Response) => {
    const { smiles } = req.body;

    if (!smiles) {
        return res.status(400).json({ error: 'SMILES string is required.' });
    }

    const options: PythonShellOptions = {
        mode: 'text',
        pythonPath: PYTHON_PATH,
        pythonOptions: ['-u'], // unbuffered binary stdout and stderr
        scriptPath: INFERENCE_SCRIPT_PATH,
        args: [smiles]
    };

    try {
        const messages = await PythonShell.run(INFERENCE_SCRIPT_NAME, options);

        // The script may output logs to stdout/stderr, but the final result is a JSON object.
        // We look for the last valid JSON object in the output stream.
        const jsonOutput = messages.find(m => m.trim().startsWith('{'));

        if (!jsonOutput) {
            throw new Error("Inference script did not return a valid JSON response.");
        }

        const result = JSON.parse(jsonOutput);
        res.json(result);

    } catch (err) {
        console.error("Inference Error:", err);

        // Return a structured error response
        // In a real production scenario, we might return a 500 here, 
        // but for the demo, we return a fallback object with a note.
        res.status(500).json({
            error: "Inference failed",
            details: err instanceof Error ? err.message : String(err)
        });
    }
});

/**
 * LAMMPS Input Generation Endpoint
 */
app.post('/api/lammps/generate', (req: Request<{}, {}, LammpsGenerateRequest>, res: Response) => {
    const { smiles, temp = 298, pressure = 1 } = req.body;

    if (!smiles) {
        return res.status(400).json({ error: 'SMILES string is required.' });
    }

    try {
        const content = generateInput(smiles, temp, pressure);
        res.json({
            filename: 'in.lammps',
            content: content
        });
    } catch (err) {
        console.error("LAMMPS Generation Error:", err);
        res.status(500).json({ error: "Failed to generate LAMMPS input." });
    }
});

/**
 * LAMMPS Log Parsing Endpoint
 */
app.post('/api/lammps/parse', (req: Request<{}, {}, LammpsParseRequest>, res: Response) => {
    const { content } = req.body;

    if (!content) {
        return res.status(400).json({ error: 'Log file content is required.' });
    }

    try {
        const result = parseLog(content);
        res.json(result);
    } catch (e) {
        console.error("Log Parsing Error:", e);
        res.status(500).json({ error: 'Failed to parse log file.' });
    }
});

// --- Server Start ---
app.listen(PORT, () => {
    console.log(`\n🚀 PolyAletheia Backend running on http://localhost:${PORT}`);
    console.log(`   Inference Script: ${path.join(INFERENCE_SCRIPT_PATH, INFERENCE_SCRIPT_NAME)}\n`);
});
