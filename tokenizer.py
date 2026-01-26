# setup tokenizer for polymer bert
from transformers import AutoTokenizer

MODEL_NAME = "answerdotai/ModernBERT-base"

def get_tokenizer():
    # load base tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # we might need to add special tokens for smiles later
    # but modernbert has a large vocab so let's see how it handles smiles first
    return tokenizer

if __name__ == "__main__":
    try:
        tok = get_tokenizer()
        print(f"Loaded tokenizer: {MODEL_NAME}")
        print(f"Vocab size: {tok.vocab_size}")
        
        # test on polystyrene
        smiles = "*CC(*)c1ccccc1"
        tokens = tok.tokenize(smiles)
        print(f"SMILES: {smiles}")
        print(f"Tokens: {tokens}")
        print(f"IDs: {tok.convert_tokens_to_ids(tokens)}")
        
    except Exception as e:
        print(f"Error: {e}")
