import torch
import numpy as np
import html
from extract_attention import extract_attention
from tokenizer import get_tokenizer

def generate_heatmap_html(smiles, attention_weights, tokens):
    # attention_weights: [seq_len] (aggregated)
    # simple scalar attention for coloring
    
    # normalize weights 0-1
    w_min, w_max = attention_weights.min(), attention_weights.max()
    norm_weights = (attention_weights - w_min) / (w_max - w_min + 1e-9)
    
    html_out = '<div style="font-family: monospace; font-size: 1.2em;">'
    
    for token, weight in zip(tokens, norm_weights):
        # calculate color (white to red)
        # alpha = weight
        # color: rgba(255, 0, 0, alpha)
        # using hsla for better look?
        # light red background
        
        bg_color = f"rgba(255, 0, 0, {weight:.2f})"
        
        # clean token
        clean_token = html.escape(token.replace("Ġ", ""))
        
        html_out += f'<span style="background-color: {bg_color}; padding: 2px;">{clean_token}</span>'
        
    html_out += "</div>"
    return html_out

def visualize_molecule(smiles):
    print(f"Visualizing: {smiles}")
    
    # get attention
    # shape: [1, 12, seq, seq]
    attn, tokens = extract_attention(smiles)
    
    # Strategy: Average over heads, maximize over source tokens?
    # We want to see "which token is most attended to?"
    # Self-attention: [target_tensor, source_tensor]
    # Sum over heads -> [seq, seq]
    # Sum over target (rows) -> "how much attention does this source token receive?"
    
    attn_avg_heads = attn[0].mean(axis=0) # [seq, seq]
    
    # Importance of token j = Sum(Attention(i -> j)) for all i
    # i.e. how much do other tokens look at j?
    token_importance = attn_avg_heads.sum(axis=0) # [seq]
    
    # remove special tokens from viz if desired?
    # keeping them for now
    
    heatmap = generate_heatmap_html(smiles, token_importance, tokens)
    
    with open("attention_heatmap.html", "w") as f:
        f.write("<html><body>")
        f.write("<h3>Attention Heatmap</h3>")
        f.write(f"<p>{smiles}</p>")
        f.write(heatmap)
        f.write("</body></html>")
        
    print("Saved attention_heatmap.html")

if __name__ == "__main__":
    test_smiles = "*CC(*)c1ccccc1"
    visualize_molecule(test_smiles)
