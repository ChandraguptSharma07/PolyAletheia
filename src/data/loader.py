# load and prep the polymer dataset
import pandas as pd
from smiles_func import is_valid_smiles, canonicalize


def load_data(path="train.csv"):
    df = pd.read_csv(path)
    return df


def clean_data(df):
    # filter out invalid smiles
    valid_mask = df["SMILES"].apply(is_valid_smiles)
    df = df[valid_mask].copy()
    
    # canonicalize
    df["SMILES"] = df["SMILES"].apply(canonicalize)
    
    return df


def get_property_stats(df):
    # check which properties have data
    props = ["Tg", "FFV", "Tc", "Density", "Rg"]
    stats = {}
    for p in props:
        non_null = df[p].notna().sum()
        stats[p] = {
            "count": non_null,
            "pct": non_null / len(df) * 100
        }
    return stats


if __name__ == "__main__":
    df = load_data()
    print(f"Loaded {len(df)} samples")
    
    df = clean_data(df)
    print(f"After cleaning: {len(df)} samples")
    
    stats = get_property_stats(df)
    for p, s in stats.items():
        print(f"{p}: {s['count']} ({s['pct']:.1f}%)")
