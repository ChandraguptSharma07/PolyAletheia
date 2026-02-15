import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from data_loader import load_data, clean_data

def create_splits(seed=42):
    print("Loading data...")
    df = load_data()
    df = clean_data(df)
    
    # we want to ensure we distribute the rare properties (Tg, Tc) fairly
    # but FFV is the most common, so let's stratify by FFV bins
    
    # drop entries with NO target properties (if any)
    # actually, all samples might be useful if we do pre-training?
    # for now, let's keep everything valid
    
    # simple random split for now to keep it simple
    # 80/10/10
    
    # create train/temp split
    train, temp = train_test_split(df, test_size=0.2, random_state=seed)
    
    # split temp into val/test
    val, test = train_test_split(temp, test_size=0.5, random_state=seed)
    
    print(f"Train: {len(train)}")
    print(f"Val:   {len(val)}")
    print(f"Test:  {len(test)}")
    
    # save them
    train.to_csv("train_split.csv", index=False)
    val.to_csv("val_split.csv", index=False)
    test.to_csv("test_split.csv", index=False)
    print("Saved split CSVs.")

if __name__ == "__main__":
    create_splits()
