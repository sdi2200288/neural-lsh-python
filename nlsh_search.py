#!/usr/bin/env python3
import argparse
import numpy as np
import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# def load_index(index_path):
#     """Φόρτωση του αποθηκευμένου ευρετηρίου"""
#     model = torch.load(f"{index_path}/model.pth")
#     model.eval()  # Σημαντικό: για inference
    
#     # Πρέπει να αποθηκεύσεις και να φορτώσεις τα inverted files
#     inv_file = np.load(f"{index_path}/inverted_file.npy", allow_pickle=True).item()
    
#     return model, inv_file

class MLPClassifier(nn.Module):
    def __init__(self, d_in, n_out, hidden=64, layers=3):
        super().__init__()
        layers_list = []
        layers_list.append(nn.Linear(d_in, hidden))
        layers_list.append(nn.ReLU())
        for _ in range(layers - 1):
            layers_list.append(nn.Linear(hidden, hidden))
            layers_list.append(nn.ReLU())
        layers_list.append(nn.Linear(hidden, n_out))
        self.net = nn.Sequential(*layers_list)

    def forward(self, x):
        return self.net(x)


def load_index(index_path):
    """Φόρτωση αποθηκευμένου ευρετηρίου Neural LSH"""

    checkpoint = torch.load(f"{index_path}/model.pth", map_location="cpu")

    # Δημιουργία ίδιου μοντέλου με αυτό που είχες στο build
    model = MLPClassifier(
        d_in = checkpoint['input_dim'],
        n_out = checkpoint['output_dim'],
        hidden = checkpoint['hidden_dim'],
        layers = checkpoint['num_layers']
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load inverted file
    inv_file = np.load(f"{index_path}/inverted_file.npy", allow_pickle=True).item()

    return model, inv_file



# -------------------------------------------------------------
# 1. LOAD DATASET (MNIST / SIFT)
# -------------------------------------------------------------
def load_dataset(path, dtype):
    print(f">> DEBUG: Loading dataset from '{path}'")
    print(f">> DEBUG: File exists: {os.path.exists(path)}")
    print(f">> DEBUG: Current directory: {os.getcwd()}")

    print(f">> Attempting to load dataset from: {path}")
    print(f">> File exists: {os.path.exists(path)}")
    
    if not os.path.exists(path):
        # Δοκίμασε να βρεις το αρχείο
        possible_paths = [
            path,
            f"../{path}",
            f"../../{path}",
            f"sift_data/{os.path.basename(path)}",
            f"../sift_data/{os.path.basename(path)}"
        ]
        for p in possible_paths:
            if os.path.exists(p):
                print(f">> DEBUG: Found file at: {p}")
                path = p
                break
        else:
            raise FileNotFoundError(f"Dataset file not found: {path}")
    if dtype == "sift":
        return load_fvecs(path)
    elif dtype == "mnist":
        return load_mnist(path)
    else:
        raise ValueError(f"Unknown dataset type: {dtype}")

def load_fvecs(path):
    """Φόρτωση SIFT dataset από δυαδικό αρχείο"""
    with open(path, "rb") as f:
        dims = np.fromfile(f, dtype=np.int32, count=1)[0]
    
    data = np.fromfile(path, dtype=np.int32)
    d = data[0]
    assert d == dims

    # Το αρχείο είναι (int32 dim + float32[d]) * N
    data = data.reshape(-1, d + 1)
    return data[:, 1:].view(np.float32)

def load_mnist(path):
    """Φόρτωση MNIST dataset"""
    # Προσαρμόστε ανάλογα με τη μορφή του MNIST dataset σας
    data = np.fromfile(path, dtype=np.uint8)
    # Υποθέτουμε 784 features per image (28x28)
    return data.reshape(-1, 784).astype(np.float32) / 255.0  # Κανονικοποίηση

def main():
    parser = argparse.ArgumentParser(description="Neural LSH Search")

    parser.add_argument("-d", dest="dataset", required=True)
    parser.add_argument("-q", dest="query_file", required=True, help="Query file")
    parser.add_argument("-i", dest="index_path", required=True, help="Index path")
    parser.add_argument("-o", dest="output_file", required=True, help="Output file")
    parser.add_argument("-type", dest="data_type", required=True, choices=["sift", "mnist"])

    # Προαιρετικά arguments
    parser.add_argument("-N", type=int, default=1, help="Number of nearest neighbors")
    parser.add_argument("-R", type=float, default=2000.0, help="Search radius")
    parser.add_argument("-T", type=int, default=5, help="Number of bins to probe")
    parser.add_argument("-range", dest="do_range_search", default="true", choices=["true", "false"])
    
    args = parser.parse_args()
        
    # STEP 1: LOAD DATASET
    print(f">> Loading dataset: {args.dataset}")
    data = load_dataset(args.dataset, args.data_type)
    n = data.shape[0]
    d = data.shape[1]
    print(f">> Dataset loaded: {n} vectors, {d} dimensions")

    print(f">> Loading queries: {args.query_file}")
    queries = load_dataset(args.query_file, args.data_type)
    print(f">> Queries loaded: {queries.shape[0]} vectors")

    # STEP 1: LOAD INDEX
    print(">> Loading Neural LSH index...")
    model, inv_file = load_index(args.index_path)

    print(">> Inverted file loaded!")
    print(f">> Total parts (m): {len(inv_file.keys())}")

    for part in sorted(inv_file.keys()):
        print(f"   Part {part}: {len(inv_file[part])} points")

    print(">> Model architecture:")
    print(model)

    # Optional: δείξε το πρώτο layer
    first_layer = model.net[0]
    print(">> First Linear layer weights shape:", first_layer.weight.shape)


if __name__ == "__main__":
    main()


# # Step 2
# logits = model(q)
# probs = softmax(logits)
# bins = np.argsort(probs)[-T:]

# # Step 3
# candidates = []
# for b in bins:
#     candidates.extend(inv_file[b])


