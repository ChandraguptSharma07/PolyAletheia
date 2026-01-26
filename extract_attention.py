import torch
import numpy as np
from model import PolymerPredictor
from tokenizer import get_tokenizer

def extract_attention(smiles, model_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # load tokenizer and model
    tokenizer = get_tokenizer()
    model = PolymerPredictor().to(device)
    
    if model_path:
        # ignore missing keys since we might load a checkpoint that has extra/less keys?
        # actually, checkpoint should match. But standard safety is good.
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded weights from {model_path}")
        except FileNotFoundError:
            print(f"Warning: {model_path} not found. Using random weights.")
    else:
        print("Using initialized weights (untrained)")
        
    model.eval()
    
    # tokenize
    encoding = tokenizer(
        smiles,
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    
    input_ids = encoding["input_ids"].to(device)
    mask = encoding["attention_mask"].to(device)
    
    # inference with attention
    with torch.no_grad():
        preds, attentions = model(input_ids, mask, output_attentions=True)
        
    # attentions is a tuple of 12 tensors (layers), each [batch, heads, seq, seq]
    # chemBerta has 6 layers usually? checking len
    print(f"Number of layers: {len(attentions)}")
    print(f"Attention shape (layer 0): {attentions[0].shape}")
    
    # return the last layer attention
    last_layer_attn = attentions[-1].cpu().numpy() # [1, 12, seq, seq]
    return last_layer_attn, encoding.tokens()

if __name__ == "__main__":
    test_smiles = "*CC(*)c1ccccc1"
    print(f"Testing attention extraction for: {test_smiles}")
    
    # try to load best_model if exists
    try:
        attn, tokens = extract_attention(test_smiles, "model.pth")
    except:
        attn, tokens = extract_attention(test_smiles)
    
    print(f"Tokens: {tokens}")
    print(f"Attention map shape: {attn.shape}")
    
    # print sample of first head
    print("Sample attention (layer -1, head 0):")
    print(np.round(attn[0, 0, :5, :5], 4))
