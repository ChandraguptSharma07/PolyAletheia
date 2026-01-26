import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import numpy as np
import pandas as pd
from tqdm import tqdm
from model import PolymerPredictor
from tokenizer import get_tokenizer

# properties we want to predict
PROPS = ["Tg", "FFV", "Tc", "Density", "Rg"]

class PolymerDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_len=128):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # normalize targets? maybe later
        # for now raw values
        self.targets = self.df[PROPS].values.astype(np.float32)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        smiles = row["SMILES"]
        
        encoding = self.tokenizer(
            smiles,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "targets": torch.tensor(self.targets[idx])
        }

def masked_mse_loss(preds, targets):
    # targets: [batch, 5], preds: [batch, 5]
    # some targets are NaN
    mask = ~torch.isnan(targets)
    
    # only calculate loss where we have ground truth
    diff = preds[mask] - targets[mask]
    loss = (diff ** 2).mean()
    
    if torch.isnan(loss):
        return torch.tensor(0.0, requires_grad=True).to(preds.device)
        
    return loss

def train(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    
    progress = tqdm(loader, desc="Training")
    for batch in progress:
        input_ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        targets = batch["targets"].to(device)
        
        optimizer.zero_grad()
        preds = model(input_ids, mask)
        
        loss = masked_mse_loss(preds, targets)
        loss.backward()
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        progress.set_postfix({"loss": loss.item()})
        
    return total_loss / len(loader)

def validate(model, loader, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            targets = batch["targets"].to(device)
            
            preds = model(input_ids, mask)
            loss = masked_mse_loss(preds, targets)
            total_loss += loss.item()
            
    return total_loss / len(loader)

if __name__ == "__main__":
    # simple training run
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    tokenizer = get_tokenizer()
    model = PolymerPredictor().to(device)
    
    train_ds = PolymerDataset("train_split.csv", tokenizer)
    val_ds = PolymerDataset("val_split.csv", tokenizer)
    
    train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=16)
    
    optimizer = AdamW(model.parameters(), lr=1e-4) # slightly high for bert but fine for test
    epochs = 2
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, len(train_dl)*epochs)
    
    for epoch in range(epochs):
        train_loss = train(model, train_dl, optimizer, scheduler, device)
        val_loss = validate(model, val_dl, device)
        print(f"Epoch {epoch+1}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}")
        
    # save
    torch.save(model.state_dict(), "model.pth")
    print("Model saved.")
