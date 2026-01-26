import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class PolymerPredictor(nn.Module):
    def __init__(self, model_name="seyonec/ChemBERTa-zinc-base-v1", num_tasks=5):
        super().__init__()
        print(f"Loading {model_name}...")
        self.backbone = AutoModel.from_pretrained(model_name)
        
        # modernbert hidden size is usually 768 for base
        hidden_size = self.backbone.config.hidden_size
        
        # simple regression head
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_tasks)
        )
        
    def forward(self, input_ids, attention_mask=None, **kwargs):
        # pass through backbone
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        
        # use CLS token representation (first token)
        # modernbert might use mean pooling, but CLS is standard for bert
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        # predict 5 properties
        predictions = self.head(cls_embedding)
        return predictions

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
