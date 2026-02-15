import torch
import numpy as np

def compute_saliency(model, tokenizer, smiles):
    """
    Computes Input x Gradient saliency for each of the 5 properties.
    Returns:
        token_importance_dict: { "Property": np.array([seq_len]) }
        tokens: list of tokens
    """
    device = next(model.parameters()).device
    model.eval()
    
    # Tokenize
    inputs = tokenizer(smiles, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
    # 1. Get Embeddings
    # We need gradients w.r.t these embeddings
    embeddings = model.get_embeddings(input_ids)
    embeddings.retain_grad() # Crucial: tell PyTorch to keep grads for this non-leaf tensor
    
    # 2. Forward pass from embeddings
    # We need to manually pass embeddings to the backbone. 
    # Standard HF models usually support 'inputs_embeds' argument.
    
    # Let's verify if we can pass inputs_embeds to model.forward
    # The model.forward calls self.backbone(...)
    # self.backbone is RobertaModel, which accepts inputs_embeds.
    
    # However, our model.forward signature asks for input_ids.
    # We should probably modify model.forward or just call backbone directly here?
    # Calling model.forward with inputs_embeds requires changing model.py to accept it.
    # Let's try to pass inputs_embeds via kwargs if model.py allows it.
    
    # Looking at model.py:
    # def forward(self, input_ids, attention_mask=None, output_attentions=False, **kwargs):
    #    outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, ...)
    
    # If we pass input_ids=None and inputs_embeds=embeddings, does it work?
    # We need to change model.py slightly to handle this standard HF pattern.
    pass

    # Actually, simpler is to just enable gradients on the input_ids? No, integer inputs don't have gradients.
    # We MUST pass inputs_embeds.
    
    # Start by computing gradients for each property head
    properties = ["Tg", "Tc", "Density", "FFV", "Rg"]
    saliency_maps = {}
    
    # We need to use a hook or modify model.py to accept inputs_embeds.
    # Let's look at model.py again. 
    # It passes **kwargs to self.backbone.
    # So if we call model(input_ids=None, inputs_embeds=embeddings), 
    # self.backbone(input_ids=None, ..., inputs_embeds=embeddings) might work.
    
    # But self.head(cls_embedding) relies on outputs.last_hidden_state
    
    try:
        # Forward pass using embeddings
        outputs = model.backbone(inputs_embeds=embeddings)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        predictions = model.head(cls_embedding) # [1, 5]
    except Exception as e:
        print(f"Error during forward pass with embeddings: {e}")
        return {}, tokens

    for i, prop_name in enumerate(properties):
        # Zero grads
        if embeddings.grad is not None:
            embeddings.grad.zero_()
            
        # Get score for this property
        score = predictions[0, i]
        
        # Backward
        score.backward(retain_graph=True)
        
        # Get gradient w.r.t embeddings
        # shape: [1, seq_len, hidden_dim]
        grads = embeddings.grad[0] 
        
        # Input x Gradient
        # shape: [seq_len, hidden_dim]
        input_x_grad = embeddings[0] * grads
        
        # Summarize to scalar per token (L2 norm or Sum)
        # Using L2 norm is common for saliency
        # shape: [seq_len]
        saliency = torch.norm(input_x_grad, dim=1).detach().cpu().numpy()
        
        
        saliency_maps[prop_name] = saliency
        
    
    # --- Contrastive Logic ---
    # Since raw saliency is highly correlated (0.99+), we subtract the mean to find differences.
    import numpy as np
    
    # PROBLEM: Magnitudes might differ significantly between heads.
    # Tg gradients were massive compared to Density, washing out the heatmap.
    # So we normalize each property's map to relative [0, 1] importance first.
    normalized_maps = []
    keys = properties # ensure order
    
    for p in keys:
        raw = saliency_maps[p]
        # Min-Max Normalize
        numerator = raw - raw.min()
        denominator = raw.max() - raw.min() + 1e-9
        norm = numerator / denominator
        normalized_maps.append(norm)
        
    # stack: [5, seq_len]
    all_saliencies = np.stack(normalized_maps)
    mean_saliency = np.mean(all_saliencies, axis=0)
    
    contrastive_maps = {}
    # print("Computing Contrastive Saliency (Diff from Normalized Mean)...") 
    
    for i, prop_name in enumerate(keys):
        # Diff from mean of NORMALIZED maps
        diff = normalized_maps[i] - mean_saliency
        
        # We only care about what is *more* important for this property than average
        contrastive_maps[prop_name] = np.maximum(diff, 0)
        
        # Scaling up for viz visibility?
        # The viz script normalizes again, so absolute scale doesn't matter, just distribution.
    
    return contrastive_maps, tokens
