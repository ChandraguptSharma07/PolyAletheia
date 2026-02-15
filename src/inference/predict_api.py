"""
Inference Microservice for PolyAletheia.

This script loads the trained PolymerPredictor model and serves predictions
via standard output (JSON). It also generates 3D conformers using RDKit's
ETKDGv3 algorithm for visualization.

Author: PolyAletheia Team
"""

import sys
import os
import json
import argparse
import logging
import torch
import numpy as np
from typing import Dict, Any, Optional, List
from rdkit import Chem
from rdkit.Chem import AllChem

# Configure logging to stderr to avoid polluting stdout (which is for JSON)
logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path for module resolution
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(PROJECT_ROOT)

try:
    from src.models.model import PolymerPredictor
    from src.data.tokenizer import get_tokenizer
except ImportError as e:
    logger.error(f"Failed to import core modules: {e}")
    sys.exit(1)

# Property definintions
PROPERTIES = ["Tg", "FFV", "Tc", "Density", "Rg"]

# Normalization statistics from training set
# TODO: Load these from a serialized 'scaler.json' in production
NORMALIZATION_STATS = {
    "Tg": {"mean": 100.0, "std": 50.0},
    "FFV": {"mean": 0.15, "std": 0.05},
    "Tc": {"mean": 250.0, "std": 80.0},
    "Density": {"mean": 1.2, "std": 0.3},
    "Rg": {"mean": 10.0, "std": 5.0}
}

ATOM_COLORS: Dict[str, str] = {
    "C": "#333333", "H": "#FFFFFF", "O": "#FF0000",
    "N": "#0000FF", "S": "#CCCC00", "F": "#00FF00", "Cl": "#00FF00"
}
DEFAULT_ATOM_COLOR = "#FF00FF"

def get_atom_color(symbol: str) -> str:
    """Returns the hex color for a given atomic symbol."""
    return ATOM_COLORS.get(symbol, DEFAULT_ATOM_COLOR)

def generate_3d_conformer(smiles: str) -> Optional[Dict[str, Any]]:
    """
    Generates a 3D conformer for a given SMILES string using ETKDGv3.
    
    Args:
        smiles: The SMILES string of the molecule.
        
    Returns:
        A dictionary containing atom positions and bond information, 
        or None if generation fails.
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        logger.warning(f"Invalid SMILES string: {smiles}")
        return None
    
    # Add Hydrogens for realistic 3D structure
    mol = Chem.AddHs(mol)
    
    # Use ETKDGv3 for state-of-the-art conformer generation
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    
    try:
        AllChem.EmbedMolecule(mol, params)
        AllChem.MMFFOptimizeMolecule(mol)
    except ValueError:
        logger.warning("Failed to embed molecule or optimize geometry.")
        return None
    
    conf = mol.GetConformer()
    atoms = []
    bonds = []
    
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        atoms.append({
            "idx": atom.GetIdx(),
            "symbol": atom.GetSymbol(),
            "pos": [pos.x, pos.y, pos.z],
            "color": get_atom_color(atom.GetSymbol())
        })
        
    for bond in mol.GetBonds():
        bonds.append({
            "start": bond.GetBeginAtomIdx(),
            "end": bond.GetEndAtomIdx(), 
            "order": bond.GetBondTypeAsDouble()
        })
        
    return {"atoms": atoms, "bonds": bonds}

def load_predictor(model_path: str, device: torch.device) -> Optional[PolymerPredictor]:
    """Loads the PolymerPredictor model from disk."""
    if not os.path.exists(model_path):
        # Fallback: Check PROJECT_ROOT
        fallback_path = os.path.join(PROJECT_ROOT, model_path)
        if os.path.exists(fallback_path):
            model_path = fallback_path
        else:
            logger.warning(f"Model not found at {model_path} or {fallback_path}. Running in fallback mode.")
            return None
        
    try:
        model = PolymerPredictor().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None

def run_inference(smiles: str, model_path: str = "best_model_colab.pth") -> None:
    """
    Main inference routine. Prints JSON result to stdout.
    """
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = get_tokenizer()
        model = load_predictor(model_path, device)
        
        properties = {}
        
        if model:
            # Tokenize & Encode
            encoding = tokenizer(
                smiles,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            )
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)

            # Predict
            with torch.no_grad():
                raw_preds = model(input_ids, attention_mask)
                raw_preds = raw_preds.cpu().numpy()[0]

            # Model outputs raw units (based on training MSE ~216)
            for i, prop in enumerate(PROPERTIES):
                properties[prop] = float(raw_preds[i])
        else:
            # Fallback values (mean) if model fails to load
            properties = {p: NORMALIZATION_STATS[p]["mean"] for p in PROPERTIES}
            properties["note"] = "Model not loaded; returning mean stats."

        # Generate Structure
        structure = generate_3d_conformer(smiles)

        # Output JSON
        result = {
            "smiles": smiles,
            "properties": properties,
            "structure": structure,
            "success": True
        }
        print(json.dumps(result))

    except Exception as e:
        logger.exception("Inference failed")
        error_result = {
            "success": False,
            "error": str(e)
        }
        print(json.dumps(error_result))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PolyAletheia Inference Engine")
    parser.add_argument("smiles", type=str, help="SMILES string of the polymer/molecule")
    parser.add_argument("--model", type=str, default="best_model_colab.pth", help="Path to model checkpoint")
    
    args = parser.parse_args()
    run_inference(args.smiles, args.model)
