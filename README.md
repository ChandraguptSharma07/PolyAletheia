# PolyAletheia - The Truth in Polymers

PolyAletheia is an AI-driven system designed to predict polymer properties (like Density) instantly using SMILES strings. But unlike black-box AI models, we prioritize **Scientific Ground Truth**.

This repository contains the full stack:
-   **AI Core**: A custom ChemBERTa + Regression Head model.
-   **Simulation Backend**: An automated LAMMPS pipeline for physical verification.
-   **"Void" UI**: A premium, minimalist React interface for exploring chemical space.

---

## 🚀 Quick Start

### 1. Prerequisites
-   Node.js & npm
-   Python 3.9+ (with `rdkit`, `torch`, `transformers`)
-   LAMMPS (for verification simulation)

### 2. Installation
```bash
# Frontend
cd frontend
npm install

# Backend
cd ../backend
npm install
```

### 3. Running the App
```bash
# Start the Backend (Port 5000)
cd backend
npm start

# Start the Frontend (Port 3000)
cd frontend
npm start
```

---

## 🛡️ Ground Truth Verification (The "Human Loop")

AI is fast, but Physics is real. We built a workflow to let you **challenge the AI** directly from the UI.

### Why Verified?
The AI predicts property $X$ in 200ms based on patterns.
LAMMPS simulates $F=ma$ for 100,000 steps to measure property $X$ from first principles.
If they agree, you know the AI is right.

### The Workflow

**1. Download the Simulation Package 📦**
On any prediction page, click **"Download Input"**.
You'll get a verified ZIP file containing:
-   `in.lammps`: Our robust simulation recipe (NPT Ensemble).
-   `data.polymer`: The exact 3D structure of your molecule (generated on-the-fly by RDKit).

**2. Run the Physics (Locally)**
Unzip the package and run LAMMPS.
*   **Windows (WSL)**: `wsl lmp -in in.lammps`
*   **Linux/Mac**: `lmp -in in.lammps`

*Note: A standard 100k step run takes about 10-20 seconds for small oligomers.*

**3. Upload the Proof 📤**
Drag and drop the resulting `log.lammps` file back into the Web App.

**4. The Verdict**
The system will parse your log, extract the *real* physical density, and compare it to the AI's prediction.
-   **Green Badge**: Verified (Error < 5%)
-   **Data Insight**: See exactly how close the AI got to reality.

---

## 🧩 Structure
-   `/src`: Python AI model & Inference scripts.
-   `/backend`: Node.js Express server (orchestrates AI & LAMMPS).
-   `/frontend`: React + Three.js + Tailwind UI.

## 🤝 Contributing
Built with 💜 by Chandragupt.
We believe in "Trust, but Verify."
