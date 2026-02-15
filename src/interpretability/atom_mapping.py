
from typing import List, Dict
import numpy as np

def map_tokens_to_atoms(smiles: str, tokens: List[str], saliency_map: np.ndarray, tokenizer) -> List[float]:
    """
    Maps token-level saliency scores to atom-level weights.
    
    Args:
        smiles: The original SMILES string.
        tokens: List of tokens from the tokenizer.
        saliency_map: Numpy array of scores parallel to tokens.
        tokenizer: The tokenizer object (for offset mapping).
        
    Returns:
        List of float weights, one for each Heavy Atom in the SMILES string.
        The order corresponds to RDKit's atom order (which matches SMILES order).
    """
    
    # 1. Get Offsets to map Tokens -> Character Spans
    try:
        encoding = tokenizer(smiles, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=True)
        offsets = encoding["offset_mapping"][0].numpy()
        # tokens list usually has special tokens <s> </s>, we need to align carefully
        # The saliency_map should be same length as input_ids
    except Exception as e:
        print(f"Mapping Error: {e}")
        return []

    # 2. Expand Token Scores to Character Scores
    # We create an array representing the importance of each character in the SMILES string
    char_scores = np.zeros(len(smiles))
    
    # We iterate through tokens/saliency.
    # Note: saliency_map length includes special tokens (start/end).
    # encoding['input_ids'] also includes them.
    # offsets for special tokens are usually (0,0).
    
    for i, (start, end) in enumerate(offsets):
        if start == end: continue # Special token or empty
        if i >= len(saliency_map): break 
        
        score = saliency_map[i]
        
        # Assign this score to all characters in the token's span
        # We take the MAX if overlaps? Usually they partition the string.
        # But let's just write valid values.
        if end > len(smiles): end = len(smiles)
        
        # Max-pooling over characters seems safer than overwriting?
        # Actually tokens partition the string roughly.
        char_scores[start:end] = np.maximum(char_scores[start:end], score)

    # 3. Map Character Scores to Atoms
    # We iterate the SMILES string and identify heavy atoms.
    # RDKit guarantees heavy atom order matches SMILES order.
    
    atom_weights = []
    i = 0
    N = len(smiles)
    
    # Simple parser to find atom starts
    # This is a heuristic but works for standard SMILES
    while i < N:
        char = smiles[i]
        
        # Check if this character starts an atom
        # Atoms: B, C, N, O, P, S, F, Cl, Br, I, etc.
        # Also inside brackets: [nH], [C@@H]
        
        is_atom = False
        atom_char_len = 0
        
        if char == '[':
            # Bracket atom (e.g., [nH]) - Treated as ONE atom unit
            # Find matching closing bracket
            j = smiles.find(']', i)
            if j != -1:
                is_atom = True
                atom_char_len = (j - i) + 1
        elif char.isalpha():
            # Potential organic subset or element
            # Check for 2-letter elements first (Cl, Br, Si, etc.)
            if i + 1 < N and smiles[i:i+2] in ["Cl", "Br", "Si", "Mg", "Na", "Li", "Zn", "Fe", "Cu", "Mn", "Co", "Ni", "Se", "Te"]:
                 is_atom = True
                 atom_char_len = 2
            # Check for 1-letter elements (B, C, N, O, P, S, F, I, b, c, n, o, p, s)
            # Note: 'r' is not an atom (part of Br/Cr/etc) unless aromatic? 'r' is not standard aromatic. 
            # 'l' is not.
            elif char in "BCNOPSFIbcnops":
                 is_atom = True
                 atom_char_len = 1
                 
        if is_atom:
            # Calculate weight for this atom
            # We take the MAX score of any character belonging to this atom
            segment_scores = char_scores[i : i + atom_char_len]
            weight = np.mean(segment_scores) if len(segment_scores) > 0 else 0.0
            atom_weights.append(float(weight))
            
            i += atom_char_len
        else:
            # Non-atom (bond, digit, parenthesis)
            i += 1
            
    return atom_weights
