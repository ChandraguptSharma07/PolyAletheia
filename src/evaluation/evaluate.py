import torch
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pandas as pd
from tqdm import tqdm
import sys
import os

# root path setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.models.model import PolymerPredictor
from src.data.tokenizer import get_tokenizer
from torch.utils.data import DataLoader, Dataset

PROPS = ["Tg", "FFV", "Tc", "Density", "Rg"]

class PolymerDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_len=128):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_len = max_len
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

def evaluate_model(model_path="best_model_colab.pth", test_csv="test_split.csv"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # load stuff
    tokenizer = get_tokenizer()
    model = PolymerPredictor().to(device)
    
    # load weights
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"Loaded weights from {model_path}")
    else:
        print(f"Error: {model_path} not found.")
        return

    model.eval()
    
    # Create dataset
    ds = PolymerDataset(test_csv, tokenizer)
    dl = DataLoader(ds, batch_size=32, shuffle=False)
    
    # storage buckets for each property
    all_preds = {p: [] for p in PROPS} # list of values
    all_targets = {p: [] for p in PROPS}
    
    with torch.no_grad():
        for batch in tqdm(dl, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            targets = batch["targets"].cpu().numpy() # [B, 5]
            
            preds = model(input_ids, mask).cpu().numpy() # [B, 5]
            
            # Unpack per property, handling NaNs
            for i, prop in enumerate(PROPS):
                # get target column for this property
                prop_targets = targets[:, i]
                prop_preds = preds[:, i]
                
                # Filter valid
                valid_mask = ~np.isnan(prop_targets)
                
                if valid_mask.any():
                    all_targets[prop].extend(prop_targets[valid_mask])
                    all_preds[prop].extend(prop_preds[valid_mask])

    print("\n--- Evaluation Results ---")
    results = []
    
    for prop in PROPS:
        y_true = np.array(all_targets[prop])
        y_pred = np.array(all_preds[prop])
        
        if len(y_true) == 0:
            print(f"{prop}: No valid data points.")
            continue
            
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        print(f"{prop:<10} | R2: {r2:.4f} | MAE: {mae:.4f} | RMSE: {rmse:.4f} | N: {len(y_true)}")
        
        results.append({
            "Property": prop,
            "R2": r2,
            "MAE": mae,
            "RMSE": rmse,
            "N": len(y_true)
        })
        
    # Save to CSV
    res_df = pd.DataFrame(results)
    res_df.to_csv("evaluation_metrics.csv", index=False)
    print("\nMetrics saved to evaluation_metrics.csv")

if __name__ == "__main__":
    evaluate_model()
