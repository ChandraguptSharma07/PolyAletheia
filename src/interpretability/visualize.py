import torch
import html
import numpy as np
import argparse
import sys
import os

# Add project root to path to ensure imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.models.model import PolymerPredictor
from transformers import AutoTokenizer
from src.interpretability.gradients import compute_saliency

# Define colors for properties
# Define colors for properties
# might need to adjust these for better contrast
PROPERTY_COLORS = {
    "Tg": (255, 0, 0),       # Red
    "Tc": (0, 128, 0),       # Green
    "Density": (128, 0, 128),# Purple
    "FFV": (0, 0, 255),      # Blue
    "Rg": (255, 165, 0)      # Orange
}
# TODO: add command line arg for custom colors?

def generate_heatmap_html(smiles, token_importance, tokens, property_name, color_rgb):
    """
    Generates an HTML block for a single property.
    """
    # normalize weights 0-1 locally for this property
    w_min, w_max = token_importance.min(), token_importance.max()
    norm_weights = (token_importance - w_min) / (w_max - w_min + 1e-9)
    
    r, g, b = color_rgb
    
    html_out = f'<div style="margin-bottom: 20px;">'
    html_out += f'<h4 style="color: rgb({r},{g},{b});">{property_name}</h4>'
    html_out += '<div style="font-family: monospace; font-size: 1.2em; border: 1px solid #ddd; padding: 10px; border-radius: 5px;">'
    
    for token, weight in zip(tokens, norm_weights):
        # alpha = weight
        bg_color = f"rgba({r}, {g}, {b}, {weight:.2f})"
        
        # clean token
        clean_token = html.escape(token.replace("Ġ", ""))
        
        # Darken text if bg is too dark? Simplified: keep black text.
        html_out += f'<span style="background-color: {bg_color}; padding: 2px; margin-right: 1px;">{clean_token}</span>'
        
    html_out += "</div></div>"
    return html_out

def visualize_molecule(smiles, model_path=None):
    print(f"Visualizing Saliency for: {smiles}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PolymerPredictor().to(device)
    
    if model_path:
        try:
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            print(f"Loaded weights from {model_path}")
        except FileNotFoundError:
            print(f"Warning: {model_path} not found. Using random weights.")
    else:
        # Auto-discover
        candidates = ["best_model_colab.pth", "best_model.pth", "model.pth"]
        found = False
        for path in candidates:
            if os.path.exists(path):
                print(f"Auto-found weights: {path}")
                model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
                found = True
                break
        if not found:
             print("No weights found. Using random weights.")

    # Retry tokenizer loading to handle network timeouts
    import time
    max_retries = 3
    for attempt in range(max_retries):
        try:
            tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
            break
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed to load tokenizer after {max_retries} attempts: {e}")
                return
            print(f"Tokenizer load failed (attempt {attempt+1}/{max_retries}). Retrying in 2s...")
            time.sleep(2)
    
    saliency_maps, tokens = compute_saliency(model, tokenizer, smiles)
    
    if not saliency_maps:
        print("Failed to compute saliency.")
        return

    # Build HTML
    full_html = "<html><body style='font-family: sans-serif; max-width: 800px; margin: 0 auto; padding: 20px;'>"
    full_html += f"<h2>Property-Specific Attribution: {smiles}</h2>"
    full_html += "<p>Method: <strong>Input x Gradient (Saliency)</strong>. Darker color = Higher influence on prediction.</p>"
    
    for prop_name, scores in saliency_maps.items():
        color = PROPERTY_COLORS.get(prop_name, (0,0,0))
        full_html += generate_heatmap_html(smiles, scores, tokens, prop_name, color)
        
    full_html += "</body></html>"
    
    with open("attention_heatmap.html", "w") as f:
        f.write(full_html)
        
    print("Saved property-specific visualization to 'attention_heatmap.html'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles", type=str, default="CC(=O)OC1=CC=CC=C1C(=O)O", help="SMILES string to visualize")
    parser.add_argument("--model_path", type=str, default=None, help="Path to model weights")
    args = parser.parse_args()
    
    visualize_molecule(args.smiles, args.model_path)
