import torch
import argparse
import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.models.model import PolymerPredictor
from src.data.tokenizer import get_tokenizer

PROPS = ["Tg", "FFV", "Tc", "Density", "Rg"]

def predict_properties(smiles, model_path="best_model_colab.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Tokenizer
    try:
        tokenizer = get_tokenizer()
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    # Load Model
    model = PolymerPredictor().to(device)
    
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            print(f"Loaded weights from {model_path}")
        except Exception as e:
            print(f"Error loading weights: {e}")
            return
    else:
        print(f"Warning: Model weights file '{model_path}' not found. Predictions will be random!")

    model.eval()

    # Preprocess Input
    encoding = tokenizer(
        smiles,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )
    
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # Inference
    print(f"\nPredicting for: {smiles}")
    with torch.no_grad():
        preds = model(input_ids, attention_mask)
        preds = preds.cpu().numpy()[0]

    # Display Results
    # kept it simple for now, just a table
    print("\n" + "="*35)
    print(f"{'Property':<10} | {'Predict':<10}")
    print("-" * 35)
    for i, prop in enumerate(PROPS):
        # rounding to 4 decimals is enough
        print(f"{prop:<10} | {preds[i]:.4f}")
    print("="*35 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict polymer properties from SMILES.")
    parser.add_argument("--smiles", type=str, default="*CC(=O)OC1=CC=CC=C1C(=O)O", help="SMILES string of the polymer")
    parser.add_argument("--model", type=str, default="best_model_colab.pth", help="Path to model weights")
    
    args = parser.parse_args()
    
    predict_properties(args.smiles, args.model)
