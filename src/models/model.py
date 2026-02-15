import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class PolymerPredictor(nn.Module):
    def __init__(self, model_name="seyonec/ChemBERTa-zinc-base-v1", num_tasks=5):
        super().__init__()
        print(f"Loading {model_name}...")
        self.backbone = AutoModel.from_pretrained(model_name, attn_implementation="eager")
        self.backbone.config.output_attentions = True
        
        # modernbert hidden size is usually 768 for base
        # might change if we swap base models
        hidden_size = self.backbone.config.hidden_size
        
        # simple regression head
        # maybe try 2 layers?
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_tasks)
        )
        
    def forward(self, input_ids=None, attention_mask=None, output_attentions=False, inputs_embeds=None, **kwargs):
        # pass through backbone
        # We allow passing either input_ids or inputs_embeds (for interpretability)
        outputs = self.backbone(
            input_ids=input_ids, 
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask, 
            output_attentions=output_attentions, 
            **kwargs
        )
        
        # use CLS token representation (first token)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        # predict 5 properties
        predictions = self.head(cls_embedding)
        
        if output_attentions:
            return predictions, outputs.attentions
            
        return predictions

    def get_embeddings(self, input_ids, **kwargs):
        """
        Helper to get embeddings for gradient-based interpretability (Saliency).
        """
        # Access the embedding layer of the backbone directly
        # ChemBERTa (RoBERTa) -> embeddings
        return self.backbone.embeddings(input_ids)

if __name__ == "__main__":
    # simple test
    model = PolymerPredictor()
    print("Model loaded.")
    
    # dummy input
    # chemBerta vocab is small, so use 100 to be safe
    input_ids = torch.randint(0, 100, (2, 10))
    
    print("Running forward pass...")
    out = model(input_ids)
    print(f"Output shape: {out.shape}") # should be [2, 5]
    print(out)
