import wandb
import sys
import os

# quick hack to run from src folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.models.model import PolymerPredictor
from src.data.tokenizer import get_tokenizer

# trying to predict these 5 props
# might add more later if dataset allows
PROPS = ["Tg", "FFV", "Tc", "Density", "Rg"]
# print(f"DEBUG: Predicting {len(PROPS)} properties: {PROPS}")

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
        
        # log to wandb
        if wandb.run is not None:
            wandb.log({"train_loss": loss.item()})
        
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
            
    val_loss = total_loss / len(loader)
    
    if wandb.run is not None:
        wandb.log({"val_loss": val_loss})
        
    return val_loss

if __name__ == "__main__":
    # simple training run
    # wandb init
    wandb.init(project="polyaletheia", mode="online") # or disabled for debugging
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    tokenizer = get_tokenizer()
    model = PolymerPredictor().to(device)
    
    # log model config
    wandb.config.update({
        "model": "ChemBERTa",
        "lr": 1e-4,
        "batch_size": 16,
        "epochs": 10
    })
    
    train_ds = PolymerDataset("train_split.csv", tokenizer)
    val_ds = PolymerDataset("val_split.csv", tokenizer)
    
    train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=16)
    
    optimizer = AdamW(model.parameters(), lr=1e-4)
    epochs = 10
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, len(train_dl)*epochs)
    
    best_val_loss = float("inf")
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        train_loss = train(model, train_dl, optimizer, scheduler, device)
        val_loss = validate(model, val_dl, device)
        print(f"Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}")
        
        # save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("Saved new best model.")
            wandb.log({"best_val_loss": best_val_loss})
        
    print("Training complete.")
    wandb.finish()
