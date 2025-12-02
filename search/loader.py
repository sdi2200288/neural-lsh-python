# Βήμα: Φωρτώνει Αρχειο από το index

import os
import torch 
import numpy as np
from ..mlp.model import MLPClassifier 

# ---------------------------
# UTIL: load index
# ---------------------------
def load_index(index_path):
    """
    Φόρτωση αποθηκευμένου ευρετηρίου Neural LSH:
      - index_path/model.pth  (saved checkpoint dict)
      - index_path/inverted_file.npy  (dict: part -> list_of_point_ids)
    """
    ckpt_path = os.path.join(index_path, "model.pth")
    inv_path = os.path.join(index_path, "inverted_file.npy")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Model checkpoint not found: {ckpt_path}")
    if not os.path.exists(inv_path):
        raise FileNotFoundError(f"Inverted file not found: {inv_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu")

    model = MLPClassifier(
        d_in = checkpoint['input_dim'],
        n_out = checkpoint['output_dim'],
        hidden = checkpoint.get('hidden_dim', 64),
        layers = checkpoint.get('num_layers', 3)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    inv_file = np.load(inv_path, allow_pickle=True).item()
    return model, inv_file

# ---------------------------
# EUCLIDEAN DISTANCE (torch)
# ---------------------------
def euclidean_distance(query_vec, candidate_vecs):
    """
    query_vec: tensor [d]
    candidate_vecs: tensor [N, d]
    returns: tensor [N] distances
    """
    return torch.norm(candidate_vecs - query_vec, dim=1)
