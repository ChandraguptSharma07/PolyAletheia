import torch
import numpy as np
import sys
import os
# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.tokenizer import get_tokenizer
from transformers import AutoTokenizer

def get_token_map(smiles):
    tokenizer = get_tokenizer()
    
    # get offsets
    encoding = tokenizer(
        smiles,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=True
    )
    
    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])
    offsets = encoding["offset_mapping"][0].numpy()
    
    print(f"SMILES: {smiles}")
    print("-" * 40)
    print(f"{'Token':<10} | {'Span':<10} | {'Content'}")
    print("-" * 40)
    
    for i, (token, (start, end)) in enumerate(zip(tokens, offsets)):
        # skip special tokens which have (0,0) or similar usually?
        # roberta uses <s> </s>
        if start == end: 
             content = "<special>"
        else:
             content = smiles[start:end]
             
        print(f"{token:<10} | {start}-{end:<8} | {content}")

if __name__ == "__main__":
    test_smiles = "*CC(*)c1ccccc1"
    get_token_map(test_smiles)
