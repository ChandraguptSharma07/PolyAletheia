# Web App Implementation Plan - PolyAletheia

## Goal
Build a Streamlit web application that:
1.  Takes a SMILES string as input.
2.  Predicts 5 polymer properties instantly.
3.  Generates and visualizes the **3D structure** of the polymer.
4.  **Heatmap Mapping**: Colors the 3D atoms based on the AI's attention/saliency (Red = High Tg impact).
5.  **Benchmarking**: Compares AI predictions vs. traditional MD simulations (LAMMPS) to highlight the speedup.

## Phase 1: Core Interface (Streamlit)
- [ ] Set up Streamlit app (`app.py`).
- [ ] Input field for SMILES string.
- [ ] Load pre-trained model (`best_model.pth`).
- [ ] Display predicted properties in a clean table/metric cards.
- [ ] **Visualize 2D Structure**: Show standard 2D molecular drawing (RDKit).

## Phase 2: 3D Visualization & Conformer Generation
- [ ] Implement `RDKit` 3D conformer generation (`AllChem.EmbedMolecule`).
- [ ] Integrate `stmol` (Streamlit Molecular Visualization) or `py3Dmol` to render the interactive 3D molecule.
- [ ] Add controls to rotate/zoom the structure.

## Phase 3: The "Bridge" (Token -> Atom Mapping) [CRITICAL]
*Challenge: The model sees tokens, but we visualize atoms.*
- [ ] Implement a robust mapper: `Token Index` -> `Atom Index`.
    - Use `tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(smiles)` to align offsets.
    - Map each token's saliency score to the atoms covered by that token.
- [ ] Aggregate scores if multiple tokens map to one atom (e.g., take the max or sum).

## Phase 4: Saliency Coloring (3D Heatmap)
- [ ] Feature: "Color by Property".
    - User selects "Tg" -> Atoms involved in rigidity (Rings) turn Red.
    - User selects "Abuse" -> Comparisons to `Density` turn Blue.
- [ ] Implement color scaling in `py3Dmol` (Pass per-atom colors to the viewer).
- [ ] **Verification**: Ensure the 3D coloring matches the 2D heatmap we already trust.

## Phase 5: LAMMPS Comparison (The "Why")
*Real-time LAMMPS is too slow (hours). We need a "Library" mode.*
- [ ] Create a "Benchmark" tab with 5-10 pre-computed polymers (e.g., Polyethylene, Polystyrene, PMMA).
- [ ] Display:
    - **Time**: "AI (0.05s)" vs "LAMMPS (48 hours)".
    - **Accuracy**: "AI Prediction: 350K" vs "LAMMPS Result: 348K".
    - **Visualization**: Show the AI's heatmap vs. the MD simulation's "Radius of Gyration" or "Free Volume" visualization.
- [ ] **Goal**: Prove that the AI is 10,000x faster with <5% error.

## Tech Stack
-   **Frontend**: Streamlit
-   **Chemistry**: RDKit, Py3Dmol
-   **Model**: PyTorch (Our trained ChemBERTa)
-   **Data**: Pre-computed LAMMPS results (JSON/CSV)
