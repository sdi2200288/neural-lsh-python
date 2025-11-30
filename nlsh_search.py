# #!/usr/bin/env python3
# import argparse
# import numpy as np
# import os
# import sys
# import time
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import DataLoader, TensorDataset

# # def load_index(index_path):
# #     """Φόρτωση του αποθηκευμένου ευρετηρίου"""
# #     model = torch.load(f"{index_path}/model.pth")
# #     model.eval()  # Σημαντικό: για inference
    
# #     # Πρέπει να αποθηκεύσεις και να φορτώσεις τα inverted files
# #     inv_file = np.load(f"{index_path}/inverted_file.npy", allow_pickle=True).item()
    
# #     return model, inv_file

# class MLPClassifier(nn.Module):
#     def __init__(self, d_in, n_out, hidden=64, layers=3):
#         super().__init__()
#         layers_list = []
#         layers_list.append(nn.Linear(d_in, hidden))
#         layers_list.append(nn.ReLU())
#         for _ in range(layers - 1):
#             layers_list.append(nn.Linear(hidden, hidden))
#             layers_list.append(nn.ReLU())
#         layers_list.append(nn.Linear(hidden, n_out))
#         self.net = nn.Sequential(*layers_list)

#     def forward(self, x):
#         return self.net(x)


# def load_index(index_path):
#     """Φόρτωση αποθηκευμένου ευρετηρίου Neural LSH"""

#     checkpoint = torch.load(f"{index_path}/model.pth", map_location="cpu")

#     # Δημιουργία ίδιου μοντέλου με αυτό που είχες στο build
#     model = MLPClassifier(
#         d_in = checkpoint['input_dim'],
#         n_out = checkpoint['output_dim'],
#         hidden = checkpoint['hidden_dim'],
#         layers = checkpoint['num_layers']
#     )

#     model.load_state_dict(checkpoint['model_state_dict'])
#     model.eval()

#     # Load inverted file
#     inv_file = np.load(f"{index_path}/inverted_file.npy", allow_pickle=True).item()

#     return model, inv_file



# # -------------------------------------------------------------
# # 1. LOAD DATASET (MNIST / SIFT)
# # -------------------------------------------------------------
# def load_dataset(path, dtype):
#     print(f">> DEBUG: Loading dataset from '{path}'")
#     print(f">> DEBUG: File exists: {os.path.exists(path)}")
#     print(f">> DEBUG: Current directory: {os.getcwd()}")

#     print(f">> Attempting to load dataset from: {path}")
#     print(f">> File exists: {os.path.exists(path)}")
    
#     if not os.path.exists(path):
#         # Δοκίμασε να βρεις το αρχείο
#         possible_paths = [
#             path,
#             f"../{path}",
#             f"../../{path}",
#             f"sift_data/{os.path.basename(path)}",
#             f"../sift_data/{os.path.basename(path)}"
#         ]
#         for p in possible_paths:
#             if os.path.exists(p):
#                 print(f">> DEBUG: Found file at: {p}")
#                 path = p
#                 break
#         else:
#             raise FileNotFoundError(f"Dataset file not found: {path}")
#     if dtype == "sift":
#         return load_fvecs(path)
#     elif dtype == "mnist":
#         return load_mnist(path)
#     else:
#         raise ValueError(f"Unknown dataset type: {dtype}")

# def load_fvecs(path):
#     """Φόρτωση SIFT dataset από δυαδικό αρχείο"""
#     with open(path, "rb") as f:
#         dims = np.fromfile(f, dtype=np.int32, count=1)[0]
    
#     data = np.fromfile(path, dtype=np.int32)
#     d = data[0]
#     assert d == dims

#     # Το αρχείο είναι (int32 dim + float32[d]) * N
#     data = data.reshape(-1, d + 1)
#     return data[:, 1:].view(np.float32)

# def load_mnist(path):
#     """Φόρτωση MNIST dataset"""
#     # Προσαρμόστε ανάλογα με τη μορφή του MNIST dataset σας
#     data = np.fromfile(path, dtype=np.uint8)
#     # Υποθέτουμε 784 features per image (28x28)
#     return data.reshape(-1, 784).astype(np.float32) / 255.0  # Κανονικοποίηση

# def main():
#     parser = argparse.ArgumentParser(description="Neural LSH Search")

#     parser.add_argument("-d", dest="dataset", required=True)
#     parser.add_argument("-q", dest="query_file", required=True, help="Query file")
#     parser.add_argument("-i", dest="index_path", required=True, help="Index path")
#     parser.add_argument("-o", dest="output_file", required=True, help="Output file")
#     parser.add_argument("-type", dest="data_type", required=True, choices=["sift", "mnist"])

#     # Προαιρετικά arguments
#     parser.add_argument("-N", type=int, default=1, help="Number of nearest neighbors")
#     parser.add_argument("-R", type=float, default=2000.0, help="Search radius")
#     parser.add_argument("-T", type=int, default=5, help="Number of bins to probe")
#     parser.add_argument("-range", dest="do_range_search", default="true", choices=["true", "false"])
    
#     args = parser.parse_args()
        
#     # STEP 1: LOAD DATASET
#     print(f">> Loading dataset: {args.dataset}")
#     data = load_dataset(args.dataset, args.data_type)
#     n = data.shape[0]
#     d = data.shape[1]
#     print(f">> Dataset loaded: {n} vectors, {d} dimensions")

#     print(f">> Loading queries: {args.query_file}")
#     queries = load_dataset(args.query_file, args.data_type)
#     print(f">> Queries loaded: {queries.shape[0]} vectors")

#     # LOAD INDEX
#     print(">> Loading Neural LSH index...")
#     model, inv_file = load_index(args.index_path)

#     print(">> Inverted file loaded!")
#     print(f">> Total parts (m): {len(inv_file.keys())}")

#     for part in sorted(inv_file.keys()):
#         print(f"   Part {part}: {len(inv_file[part])} points")

#     # print(">> Model architecture:")
#     # print(model)

#     # Optional: δείξε το πρώτο layer
#     first_layer = model.net[0]
#     print(">> First Linear layer weights shape:", first_layer.weight.shape)

#     # ΠΡΟΣΘΗΚΗ: Έλεγχος softmax
#     print("\n>> Softmax verification for first query:")

#     # Για καθε query
#     for query_idx, query_vector in enumerate(queries):
#         if query_idx >= 1:  # Μόνο για το πρώτο query για testing
#             break

#         # Softmax 
#         query_tensor = torch.tensor(query_vector, dtype=torch.float32).unsqueeze(0)
#         with torch.no_grad():
#             raw_output = model(query_tensor)    # fθ(q)
#             probabilities = F.softmax(raw_output, dim=1).squeeze(0) # M(q) = softmax(fθ(q))
        
#         # ΕΛΕΓΧΟΙ:
#         print(f"Query {query_idx}:")
#         print(f"  Raw output shape: {raw_output.shape}")
#         print(f"  Probabilities shape: {probabilities.shape}")
#         print(f"  Raw output (first 5): {raw_output[0][:5].numpy()}")
#         print(f"  Probabilities (first 5): {probabilities[:5].numpy()}")
        
#         # Ελεγχοι softmax:
#         print(f"  Sum of all probabilities: {probabilities.sum().item():.6f}") 
#         print(f"  All probabilities >= 0: {torch.all(probabilities >= 0).item()}")
#         print(f"  Max probability: {probabilities.max().item():.4f}")
#         print(f"  Min probability: {probabilities.min().item():.4f}")
        
#         # STEP 2: bins
#         T = min(args.T, len(probabilities))
#         top_probs, top_indices = torch.topk(probabilities, T)
#         print(f"  Top-{T} bins: {top_indices.numpy()}")
#         print(f"  Top-{T} probabilities: {top_probs.numpy()}")
#         print(f"  Sum of top-{T} probabilities: {top_probs.sum().item():.4f}")

#         top_bins = top_indices.numpy()  
#         print(f"  Selected top-{T} bins: {top_bins}")
        
#         # STEP 3: Συλλογη υποψηφιων
#         candidates = set()
#         for bin_id in top_bins:
#             if bin_id in inv_file:
#                 candidates.update(inv_file[bin_id])
#         candidates = list(candidates)
#         print(f"  Found {len(candidates)} candidate points")

# if __name__ == "__main__":
#     main()


# #!/usr/bin/env python3
# import argparse
# import numpy as np
# import os
# import sys
# import time
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import DataLoader, TensorDataset

# class MLPClassifier(nn.Module):
#     def __init__(self, d_in, n_out, hidden=64, layers=3):
#         super().__init__()
#         layers_list = []
#         layers_list.append(nn.Linear(d_in, hidden))
#         layers_list.append(nn.ReLU())
#         for _ in range(layers - 1):
#             layers_list.append(nn.Linear(hidden, hidden))
#             layers_list.append(nn.ReLU())
#         layers_list.append(nn.Linear(hidden, n_out))
#         self.net = nn.Sequential(*layers_list)

#     def forward(self, x):
#         return self.net(x)


# def load_index(index_path):
#     """Φόρτωση αποθηκευμένου ευρετηρίου Neural LSH"""

#     checkpoint = torch.load(f"{index_path}/model.pth", map_location="cpu")

#     # Δημιουργία ίδιου μοντέλου με αυτό που είχες στο build
#     model = MLPClassifier(
#         d_in = checkpoint['input_dim'],
#         n_out = checkpoint['output_dim'],
#         hidden = checkpoint['hidden_dim'],
#         layers = checkpoint['num_layers']
#     )

#     model.load_state_dict(checkpoint['model_state_dict'])
#     model.eval()

#     # Load inverted file
#     inv_file = np.load(f"{index_path}/inverted_file.npy", allow_pickle=True).item()

#     return model, inv_file


# # -------------------------------------------------------------
# # EUCLIDEAN DISTANCE FUNCTION
# # -------------------------------------------------------------
# def euclidean_distance(query_vec, candidate_vecs):
#     """
#     Υπολογίζει L2 distances:
#     query_vec: [d]
#     candidate_vecs: [N, d]
#     return: [N]
#     """
#     return torch.norm(candidate_vecs - query_vec, dim=1)


# # -------------------------------------------------------------
# # LOAD DATASET FUNCTIONS
# # -------------------------------------------------------------
# def load_dataset(path, dtype):
#     print(f">> DEBUG: Loading dataset from '{path}'")
#     print(f">> DEBUG: File exists: {os.path.exists(path)}")
#     print(f">> DEBUG: Current directory: {os.getcwd()}")

#     print(f">> Attempting to load dataset from: {path}")
#     print(f">> File exists: {os.path.exists(path)}")
    
#     if not os.path.exists(path):
#         # Δοκίμασε να βρεις το αρχείο
#         possible_paths = [
#             path,
#             f"../{path}",
#             f"../../{path}",
#             f"sift_data/{os.path.basename(path)}",
#             f"../sift_data/{os.path.basename(path)}"
#         ]
#         for p in possible_paths:
#             if os.path.exists(p):
#                 print(f">> DEBUG: Found file at: {p}")
#                 path = p
#                 break
#         else:
#             raise FileNotFoundError(f"Dataset file not found: {path}")
#     if dtype == "sift":
#         return load_fvecs(path)
#     elif dtype == "mnist":
#         return load_mnist(path)
#     else:
#         raise ValueError(f"Unknown dataset type: {dtype}")

# def load_fvecs(path):
#     """Φόρτωση SIFT dataset από δυαδικό αρχείο"""
#     with open(path, "rb") as f:
#         dims = np.fromfile(f, dtype=np.int32, count=1)[0]
    
#     data = np.fromfile(path, dtype=np.int32)
#     d = data[0]
#     assert d == dims

#     # Το αρχείο είναι (int32 dim + float32[d]) * N
#     data = data.reshape(-1, d + 1)
#     return data[:, 1:].view(np.float32)

# def load_mnist(path):
#     """Φόρτωση MNIST dataset"""
#     # Προσαρμόστε ανάλογα με τη μορφή του MNIST dataset σας
#     data = np.fromfile(path, dtype=np.uint8)
#     # Υποθέτουμε 784 features per image (28x28)
#     return data.reshape(-1, 784).astype(np.float32) / 255.0  # Κανονικοποίηση


# # -------------------------------------------------------------
# # MAIN
# # -------------------------------------------------------------
# def main():
#     parser = argparse.ArgumentParser(description="Neural LSH Search")

#     parser.add_argument("-d", dest="dataset", required=True)
#     parser.add_argument("-q", dest="query_file", required=True, help="Query file")
#     parser.add_argument("-i", dest="index_path", required=True, help="Index path")
#     parser.add_argument("-o", dest="output_file", required=True, help="Output file")
#     parser.add_argument("-type", dest="data_type", required=True, choices=["sift", "mnist"])

#     # Προαιρετικά arguments
#     parser.add_argument("-N", type=int, default=1, help="Number of nearest neighbors")
#     parser.add_argument("-R", type=float, default=2000.0, help="Search radius")
#     parser.add_argument("-T", type=int, default=5, help="Number of bins to probe")
#     parser.add_argument("-range", dest="do_range_search", default="true", choices=["true", "false"])
    
#     args = parser.parse_args()
        
#     # STEP 1: LOAD DATASET
#     print(f">> Loading dataset: {args.dataset}")
#     data = load_dataset(args.dataset, args.data_type)
#     n = data.shape[0]
#     d = data.shape[1]
#     print(f">> Dataset loaded: {n} vectors, {d} dimensions")

#     print(f">> Loading queries: {args.query_file}")
#     queries = load_dataset(args.query_file, args.data_type)
#     print(f">> Queries loaded: {queries.shape[0]} vectors")

#     # LOAD INDEX
#     print(">> Loading Neural LSH index...")
#     model, inv_file = load_index(args.index_path)

#     print(">> Inverted file loaded!")
#     print(f">> Total parts (m): {len(inv_file.keys())}")

#     for part in sorted(inv_file.keys()):
#         print(f"   Part {part}: {len(inv_file[part])} points")

#     # Optional: δείξε το πρώτο layer
#     first_layer = model.net[0]
#     print(">> First Linear layer weights shape:", first_layer.weight.shape)

#     # -------------------------------------------------------------
#     # EXACT SEARCH IMPLEMENTATION
#     # -------------------------------------------------------------
#     print("\n>> Starting Exact Search...")
#     results = []

#     for qid, qvec in enumerate(queries):
#         print(f"\n>> Processing Query {qid}...")

#         q = torch.tensor(qvec, dtype=torch.float32).unsqueeze(0)

#         # Neural hashing
#         with torch.no_grad():
#             raw_output = model(q)
#             probs = F.softmax(raw_output, dim=1).squeeze(0)

#         # SOFTMAX VERIFICATION (από το δεύτερο αρχείο)
#         if qid < 2:  # Print only for first 2 queries for debugging
#             print(f"  Query {qid}:")
#             print(f"    Raw output shape: {raw_output.shape}")
#             print(f"    Probabilities shape: {probs.shape}")
#             print(f"    Raw output (first 5): {raw_output[0][:5].numpy()}")
#             print(f"    Probabilities (first 5): {probs[:5].numpy()}")
#             print(f"    Sum of all probabilities: {probs.sum().item():.6f}") 
#             print(f"    All probabilities >= 0: {torch.all(probs >= 0).item()}")
#             print(f"    Max probability: {probs.max().item():.4f}")
#             print(f"    Min probability: {probs.min().item():.4f}")

#         T = min(args.T, len(probs))
#         top_probs, top_bins = torch.topk(probs, T)
#         top_bins = top_bins.numpy()

#         # DEBUG INFO (από το δεύτερο αρχείο)
#         if qid < 2:
#             print(f"    Top-{T} bins: {top_bins}")
#             print(f"    Top-{T} probabilities: {top_probs.numpy()}")
#             print(f"    Sum of top-{T} probabilities: {top_probs.sum().item():.4f}")

#         # Collect candidates
#         candidates = set()
#         for bin_id in top_bins:
#             if bin_id in inv_file:
#                 candidates.update(inv_file[bin_id])
#         candidates = list(candidates)

#         if qid < 2:
#             print(f"    Found {len(candidates)} candidate points")

#         if len(candidates) == 0:
#             results.append((qid, []))
#             continue

#         # Compute L2 distances via the SEPARATE FUNCTION
#         cand_vecs = torch.tensor(data[candidates], dtype=torch.float32)
#         q_squeezed = q.squeeze(0)

#         dists = euclidean_distance(q_squeezed, cand_vecs)

#         # Range or Top-N
#         if args.do_range_search == "true":
#             R = args.R
#             mask = dists <= R
#             final_ids = [candidates[i] for i in range(len(candidates)) if mask[i]]
#             if qid < 2:
#                 print(f"    Range search (R={R}): found {len(final_ids)} results")
#         else:
#             N = min(args.N, len(candidates))
#             _, top_idx = torch.topk(-dists, N)
#             final_ids = [candidates[i] for i in top_idx.numpy()]
#             if qid < 2:
#                 print(f"    Top-{N} search: found {len(final_ids)} results")

#         results.append((qid, final_ids))

#         # Show progress every 100 queries
#         if (qid + 1) % 100 == 0:
#             print(f">> Processed {qid + 1} queries...")

#     # OUTPUT
#     print(f"\n>> Writing results to {args.output_file}")
#     with open(args.output_file, "w") as f:
#         for qid, ids in results:
#             f.write(f"{qid}: " + " ".join(map(str, ids)) + "\n")

#     print(">> Search completed successfully.")


# if __name__ == "__main__":
#     main()




#!/usr/bin/env python3
import argparse
import numpy as np
import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# MODEL
# ---------------------------
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

# ---------------------------
# DATA LOADING
# ---------------------------
def load_dataset(path, dtype):
    if not os.path.exists(path):
        # try some relative locations
        possible = [
            path,
            os.path.join("sift_data", os.path.basename(path)),
            os.path.join("..", path),
            os.path.join("..", "sift_data", os.path.basename(path))
        ]
        found = False
        for p in possible:
            if os.path.exists(p):
                path = p
                found = True
                break
        if not found:
            raise FileNotFoundError(f"Dataset file not found: {path}")

    if dtype == "sift":
        return load_fvecs(path)
    elif dtype == "mnist":
        return load_mnist(path)
    else:
        raise ValueError(f"Unknown dataset type: {dtype}")

def load_fvecs(path):
    """
    φορτώνει αρχείο fvecs-like όπως στην εκφώνηση
    format: (int32 dim) + float32[d] repeated
    """
    with open(path, "rb") as f:
        dims = np.fromfile(f, dtype=np.int32, count=1)[0]
    data = np.fromfile(path, dtype=np.int32)
    d = data[0]
    assert d == dims, "Mismatch dims reading fvecs-like file"
    data = data.reshape(-1, d + 1)
    # return as float32 (view)
    return data[:, 1:].view(np.float32)

def load_mnist(path):
    """
    φορτώνει mnist-like αρχεία (raw uint8)
    υποθέτουμε 784 per image
    """
    data = np.fromfile(path, dtype=np.uint8)
    return data.reshape(-1, 784).astype(np.float32) / 255.0

# ---------------------------
# TRUE exhaustive neighbors (numpy for whole-dataset)
# ---------------------------
def compute_true_neighbors_numpy(qvec, data, N):
    """
    qvec: numpy [d]
    data: numpy [n, d]
    returns: (top_indices [N], top_dists [N])
    """
    # vectorized squared distances for speed
    diff = data - qvec  # [n, d]
    sq = np.sum(diff * diff, axis=1)
    # get smallest N
    if N >= len(sq):
        idx = np.argsort(sq)
    else:
        idx = np.argpartition(sq, N-1)[:N]
        idx = idx[np.argsort(sq[idx])]
    dists = np.sqrt(sq[idx])
    return idx, dists

# ---------------------------
# MAIN
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Neural LSH Search")
    parser.add_argument("-d", dest="dataset", required=True)
    parser.add_argument("-q", dest="query_file", required=True)
    parser.add_argument("-i", dest="index_path", required=True)
    parser.add_argument("-o", dest="output_file", required=True)
    parser.add_argument("-type", dest="data_type", required=True, choices=["sift", "mnist"])
    parser.add_argument("-N", type=int, default=1)
    parser.add_argument("-R", type=float, default=2000.0)
    parser.add_argument("-T", type=int, default=5)
    parser.add_argument("-range", dest="do_range_search", default="true", choices=["true", "false"])
    args = parser.parse_args()

    # load dataset & queries
    print(f">> Loading dataset: {args.dataset}")
    data = load_dataset(args.dataset, args.data_type)
    n, d = data.shape
    print(f">> Dataset loaded: {n} vectors, {d} dimensions")

        
    # -------------------------------------------
    # LIMIT SEARCH TO FIRST 70,000 SIFT VECTORS
    # -------------------------------------------
    MAX_POINTS = 70000
    if n > MAX_POINTS:
        print(f">> Limiting database from {n} to {MAX_POINTS} vectors")
        data = data[:MAX_POINTS]
        n = MAX_POINTS
    # -------------------------------------------


    print(f">> Loading queries: {args.query_file}")
    queries = load_dataset(args.query_file, args.data_type)
    Q = queries.shape[0]
    print(f">> Queries loaded: {Q} vectors")

    # load index
    print(">> Loading Neural LSH index...")
    model, inv_file = load_index(args.index_path)
    print(">> Inverted file loaded!")
    print(f">> Total parts (m): {len(inv_file.keys())}")
    # print distribution (optional)
    for p in sorted(inv_file.keys())[:10]:
        print(f"   Part {p}: {len(inv_file[p])} points")
    # stats
    total_approx_time = 0.0
    total_true_time = 0.0

    method_name = "Neural LSH"

    results_per_query = []  # store for potential further use

    # open output file and write per query
    with open(args.output_file, "w") as fout:

        for qid, qvec in enumerate(queries):
            # ---------- approximate (Neural LSH multi-probe) ----------
            t0_approx = time.time()

            q_t = torch.tensor(qvec, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits = model(q_t)  # [1, m]
                probs = F.softmax(logits, dim=1).squeeze(0)  # [m]

            T = min(args.T, probs.shape[0])
            top_probs, top_bins = torch.topk(probs, T)
            top_bins = top_bins.numpy()

            # collect candidates from inverted file
            candidates = set()
            for b in top_bins:
                if b in inv_file:
                    candidates.update(inv_file[b])
            candidates = list(candidates)

            approx_ids = []
            approx_dists = np.array([])  # distances for reported results

            if len(candidates) > 0:
                cand_vecs = torch.tensor(data[candidates], dtype=torch.float32)
                q_single = q_t.squeeze(0)
                dists = euclidean_distance(q_single, cand_vecs)  # tensor [Ncand]
                # range vs top-N
                if args.do_range_search == "true":
                    R = args.R
                    mask = (dists <= R)
                    mask_np = mask.numpy()
                    final_ids = [candidates[i] for i in range(len(candidates)) if mask_np[i]]
                    final_dists = dists.numpy()[mask_np]
                else:
                    Nreq = min(args.N, len(candidates))
                    # topk smallest distances -> use negative trick
                    _, top_idx = torch.topk(-dists, Nreq)
                    top_idx = top_idx.numpy()
                    final_ids = [candidates[i] for i in top_idx]
                    final_dists = dists.numpy()[top_idx]
                approx_ids = final_ids
                approx_dists = final_dists
            else:
                approx_ids = []
                approx_dists = np.array([])

            t1_approx = time.time()
            t_approx = t1_approx - t0_approx
            total_approx_time += t_approx

            # ---------- true exhaustive search ----------
            t0_true = time.time()
            true_topN_idx, true_topN_dists = compute_true_neighbors_numpy(qvec, data, args.N)
            t1_true = time.time()
            t_true = t1_true - t0_true
            total_true_time += t_true

            # ---------- metrics ----------
            # Recall@N: fraction of true_topN indices found in approx_ids
            true_set = set(true_topN_idx.tolist())
            approx_set = set(approx_ids)
            if args.N > 0:
                recall = len(true_set.intersection(approx_set)) / float(args.N)
            else:
                recall = 0.0

            # Average Approximation Factor (AF): we compute for each reported approx id:
            # AF_i = true_distance(approx_id) / true_distance_of_true_best_neighbor
            # then average over reported ids. If no reported ids, AF = 1.0
            if len(approx_ids) > 0:
                best_true_dist = true_topN_dists[0] if len(true_topN_dists) > 0 else 1e-12
                af_values = []
                for aid in approx_ids:
                    # true distance of that aid (compute quickly via numpy)
                    dist_a = np.linalg.norm(data[aid] - qvec)
                    af_values.append(dist_a / (best_true_dist + 1e-12))
                avg_af = float(np.mean(af_values))
            else:
                avg_af = 1.0

            # QPS: number of queries processed per second using (approx + true) time per query
            total_time_for_query = t_approx + t_true
            qps = 1.0 / total_time_for_query if total_time_for_query > 0 else float("inf")

            # ---------- write output in required format ----------
            # Method name
            fout.write(f"{method_name}\n")
            fout.write(f"Query: {qid}\n")

            # If we return Top-N neighbors (ordered), write them
            # For range search, we still list the returned neighbors (order arbitrary)
            if len(approx_ids) > 0:
                for rank, aid in enumerate(approx_ids):
                    fout.write(f"Nearest neighbor-{rank+1}: {aid}\n")
                    # approximate distance (distance computed on candidate set; but it's Euclidean so is exact for that pair)
                    # get approx distance
                    ad = float(approx_dists[rank]) if rank < len(approx_dists) else float(np.linalg.norm(data[aid] - qvec))
                    fout.write(f"distanceApproximate: {ad:.6f}\n")
                    # true distance: compute (exact over whole dataset)
                    td = float(np.linalg.norm(data[aid] - qvec))
                    fout.write(f"distanceTrue: {td:.6f}\n")
            else:
                # no neighbors returned
                fout.write(f"Nearest neighbor-1: -1\n")
                fout.write(f"distanceApproximate: {0.0:.6f}\n")
                fout.write(f"distanceTrue: {0.0:.6f}\n")

            # R-near neighbors (if range mode)
            if args.do_range_search == "true":
                fout.write("R-near neighbors:\n")
                if len(approx_ids) > 0:
                    for aid in approx_ids:
                        fout.write(f"{aid}\n")
                # else leave empty block (just header)

            # Metrics block
            fout.write(f"Average AF: {avg_af:.6f}\n")
            fout.write(f"Recall@N: {recall:.6f}\n")
            fout.write(f"QPS: {qps:.6f}\n")

            # average times up to this query:
            processed = qid + 1
            fout.write(f"tApproximateAverage: { (total_approx_time / processed):.6f}\n")
            fout.write(f"tTrueAverage: { (total_true_time / processed):.6f}\n")
            fout.write("\n")

            # keep results if needed later
            results_per_query.append({
                "qid": qid,
                "approx_ids": approx_ids,
                "approx_dists": approx_dists.tolist() if isinstance(approx_dists, np.ndarray) else [],
                "true_topN_idx": true_topN_idx.tolist(),
                "true_topN_dists": true_topN_dists.tolist(),
                "recall": recall,
                "avg_af": avg_af,
                "qps": qps,
                "t_approx": t_approx,
                "t_true": t_true
            })

            # occasional progress print
            if (qid + 1) % 50 == 0 or (qid + 1) == Q:
                print(f">> Processed {qid+1}/{Q} queries. Avg tApprox: {total_approx_time/(qid+1):.4f}s, Avg tTrue: {total_true_time/(qid+1):.4f}s")

    
    # -------------------------------------------
    # WRITE TOTAL METRICS (AFTER ALL QUERIES)
    # -------------------------------------------

    avg_recall_all = np.mean([r["recall"] for r in results_per_query])
    avg_af_all = np.mean([r["avg_af"] for r in results_per_query])
    avg_qps_all = np.mean([r["qps"] for r in results_per_query])
    avg_tapprox_all = total_approx_time / Q
    avg_ttrue_all = total_true_time / Q

    with open(args.output_file, "a") as fout:
        fout.write("===== TOTAL RESULTS =====\n")
        fout.write(f"Total Queries: {Q}\n")
        fout.write(f"Database Size: {n}\n")
        fout.write(f"Average AF (all queries): {avg_af_all:.6f}\n")
        fout.write(f"Average Recall@{args.N}: {avg_recall_all:.6f}\n")
        fout.write(f"Average QPS: {avg_qps_all:.6f}\n")
        fout.write(f"Average tApproximate: {avg_tapprox_all:.6f}\n")
        fout.write(f"Average tTrue: {avg_ttrue_all:.6f}\n")
        fout.write("=========================\n\n")

    print(">> Search completed successfully.")
    print(f">> Results written to: {args.output_file}")

if __name__ == "__main__":
    main()

