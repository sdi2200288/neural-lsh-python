# #!/usr/bin/env python3
# import argparse
# from kahip import kaffpa
# import subprocess
# import numpy as np
# import os
# import networkx as nx
# import matplotlib.pyplot as plt
# import sys


# # Ανακατεύθυνση stdout σε αρχείο
# class OutputLogger:
#     def __init__(self, filename):
#         self.terminal = sys.stdout
#         self.log = open(filename, "w", encoding="utf-8")
    
#     def write(self, message):
#         self.terminal.write(message)
#         self.log.write(message)
#         self.log.flush()
    
#     def flush(self):
#         self.terminal.flush()
#         self.log.flush()
    
#     def close(self):
#         self.log.close()

# # -------------------------------------------------------------
# # 1. LOAD DATASET (MNIST / SIFT)
# # -------------------------------------------------------------
# def load_dataset(path, dtype):
#     if dtype == "sift":
#         return load_fvecs(path)
#     # else:
#     #     return load_mnist(path)

# def load_fvecs(path):
#     with open(path, "rb") as f:
#         dims = np.fromfile(f, dtype=np.int32, count=1)[0]
    
#     data = np.fromfile(path, dtype=np.int32)
#     d = data[0]
#     assert d == dims

#     # Το αρχείο είναι (int32 dim + float32[d]) * N
#     # Άρα κάθε vector έχει d+1 στοιχεία 4-byte
#     data = data.reshape(-1, d + 1)

#     # Μετατρέπουμε μόνο τα floats
#     return data[:, 1:].view(np.float32)

# # -------------------------------------------------------------
# # 2. CALL THE C++ ivfflat_knn EXECUTABLE TO GENERATE GRAPH
# # -------------------------------------------------------------
# def build_knn_graph(args):
#     print(">> Building k-NN graph with IVFFlat executable...")

#     cmd = [
#         "./../ergasia1/Project/search",
#         "-ivfflat_knn",
#         "-d", args.dataset,
#         "-type", args.type,
#         "--knn", str(args.knn),
#         "-kclusters", "14",
#         "-nprobe", "2",
#         "-o", "knn_output.txt"
#     ]

#     subprocess.run(cmd, check=True)

#     if not os.path.exists("knn_output.txt"):
#         raise RuntimeError("knn_output.txt was not created. Check that the executable ran correctly.")
    
#     print(">> First 10 lines of knn_output.txt:")
#     with open("knn_output.txt") as f:
#         for i, line in enumerate(f):
#             if i < 10:
#                 print(f"Line {i}: {line.strip()}")
#             else:
#                 break 
#     print(">> Finished running IVFFlat, reading knn_output.txt")
#     return read_knn_graph("knn_output.txt")

# def read_knn_graph(filename):
#     graph = {}
#     with open(filename) as f:
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue
#             node, neighs = line.split(":")
#             node = int(node)
#             neigh_list = neighs.strip().split()
#             graph[node] = [int(n) for n in neigh_list]
#     return graph

# # -------------------------------------------------------------
# # 3. MAKE GRAPH UNDIRECTED + ADD WEIGHTS (1 or 2)
# # -------------------------------------------------------------
# def make_weighted_undirected(graph):
#     """Δημιουργία συμμετρικού γράφου με ΣΩΣΤΑ βάρη"""
#     # Πρώτα, αφαίρεση self-loops και διπλοεγγραφών
#     cleaned_graph = {}
#     for u in graph:
#         # Αφαίρεση self-loops και διπλοεγγραφών, διατήρηση μόνο μοναδικών γειτόνων
#         unique_neighbors = list(set([nb for nb in graph[u] if nb != u]))
#         cleaned_graph[u] = unique_neighbors
    
#     # Υπολογισμός βαρών
#     edge_weights = {}
#     for u in cleaned_graph:
#         for v in cleaned_graph[u]:
#             edge = tuple(sorted((u, v)))
#             edge_weights[edge] = edge_weights.get(edge, 0) + 1
    
#     # Δημιουργία τελικού συμμετρικού γράφου
#     adj = {}
#     for (u, v), count in edge_weights.items():
#         weight = 2 if count == 2 else 1
        
#         if u not in adj:
#             adj[u] = []
#         if v not in adj:
#             adj[v] = []
            
#         # Προσθήκη και από τις δύο πλευρές με το ΙΔΙΟ βάρος
#         adj[u].append((v, weight))
#         adj[v].append((u, weight))
    
#     return adj

# def check_graph_consistency(adj):
#     """Έλεγχος ότι ο γράφος είναι συμμετρικός"""
#     print(">> Checking graph consistency...")
    
#     inconsistencies = 0
#     for u in adj:
#         for (v, w1) in adj[u]:
#             # Βρες το αντίστροφο edge
#             found = False
#             reverse_weight = None
#             if v in adj:
#                 for (u2, w2) in adj[v]:
#                     if u2 == u:
#                         found = True
#                         reverse_weight = w2
#                         break
            
#             if not found:
#                 print(f"ERROR: Edge {u}→{v} exists but {v}→{u} missing!")
#                 inconsistencies += 1
#             elif w1 != reverse_weight:
#                 print(f"ERROR: Edge {u}→{v} has weight {w1} but {v}→{u} has weight {reverse_weight}")
#                 inconsistencies += 1
    
#     if inconsistencies == 0:
#         print("✓ Graph is perfectly symmetric!")
#     else:
#         print(f"✗ Found {inconsistencies} inconsistencies")
    
#     return inconsistencies == 0

# def visualize_graph(adj, max_nodes=20):
#     G = nx.Graph()
#     for u, neighbors in adj.items():
#         for v, w in neighbors:
#             G.add_edge(u, v, weight=w)
#     plt.figure(figsize=(10, 10))
#     pos = nx.spring_layout(G)
#     nx.draw(G, pos, with_labels=True, node_size=300, node_color="skyblue")
#     edge_labels = nx.get_edge_attributes(G, 'weight')
#     nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
#     plt.savefig('graph_visualization.png')
#     print(">> Graph visualization saved as 'graph_visualization.png'")
#     plt.close()

# # 4. CONVERT TO CSR FOR KAHIP
# def to_csr(adj, n):
#     print(">> Converting graph to CSR format...")
#     xadj = [0]
#     adjncy = []
#     adjcwgt = []
#     vwgt = [1] * n  # all vertex weights = 1

#     for i in range(n):
#         neighbors = adj.get(i, [])
#         for (v, w) in neighbors:
#             adjncy.append(v)
#             adjcwgt.append(w)
#         xadj.append(len(adjncy))

#     return vwgt, xadj, adjncy, adjcwgt

# # -------------------------------------------------------------
# # MAIN FUNCTION
# # -------------------------------------------------------------
# def main():
#     parser = argparse.ArgumentParser()

#     parser.add_argument("-d", dest="dataset", required=True)
#     parser.add_argument("-i", dest="index", required=True)
#     parser.add_argument("-type", dest="type", required=True)

#     parser.add_argument("--knn", type=int, default=10)
#     parser.add_argument("-m", type=int, default=100)
#     parser.add_argument("--imbalance", type=float, default=0.03)
#     parser.add_argument("--kahip_mode", type=int, default=2)

#     parser.add_argument("--layers", type=int, default=3)
#     parser.add_argument("--nodes", type=int, default=64)
#     parser.add_argument("--epochs", type=int, default=10)
#     parser.add_argument("--batch_size", type=int, default=128)
#     parser.add_argument("--lr", type=float, default=0.001)

#     parser.add_argument("--seed", type=int, default=1)

#     args = parser.parse_args()

#     # Ανακατεύθυνση output σε αρχείο
#     output_file = "output.txt"
#     output_logger = OutputLogger(output_file)
#     sys.stdout = output_logger
    
#     try:
#         print("=" * 60)
#         print("KAHIP PARTITIONING PROCESS STARTED")
#         print("=" * 60)
        
#         # STEP 1: LOAD DATASET
#         print(f">> Loading dataset: {args.dataset}")
#         data = load_dataset(args.dataset, args.type)
#         n = data.shape[0]
#         print(f">> Dataset loaded: {n} vectors")

#         # STEP 2: BUILD kNN GRAPH USING C++
#         knn_graph = build_knn_graph(args)
#         print(f">> k-NN graph built with {len(knn_graph)} nodes")

#         # STEP 3: PREPARE GRAPH FOR KAHIP 
#         print(">> Making graph undirected and weighted...")
#         adj = make_weighted_undirected(knn_graph)

#         check_graph_consistency(adj)
        
#         print(">> Graph structure summary:")
#         print(f"   Total nodes: {len(adj)}")
#         total_edges = sum(len(neighbors) for neighbors in adj.values()) // 2
#         print(f"   Total edges: {total_edges}")
        
#         print(">> Example of k-NN graph (first 5 nodes):")
#         for i, (node, neighbors) in enumerate(adj.items()):
#             if i >= 5:
#                 break
#             neigh_str = ", ".join([f"{v}(w={w})" for v, w in neighbors[:10]])
#             print(f"   Node {node}: {neigh_str}")

#         # Αποθήκευση γραφικής αναπαράστασης
#         visualize_graph(adj)

#         # Χρησιμοποιούμε τον πραγματικό αριθμό κόμβων του γράφου, όχι του dataset!
#         n_nodes = len(adj)
#         vwgt, xadj, adjncy, adjcwgt = to_csr(adj, n_nodes)

#         # ---------------------------------------------------------
#         # STEP 4: RUN KAHIP PARTITIONING
#         # ---------------------------------------------------------
#         print(">> Running KaHIP...")

#         try:
#             edgecut, partition = kaffpa(
#                 vwgt,          # σωστό
#                 xadj,          # σωστό
#                 adjncy,       # ΣΩΣΤΟ - ΠΡΕΠΕΙ ΝΑ ΕΙΝΑΙ 3ο
#                 adjcwgt,        # ΣΩΣΤΟ - ΠΡΕΠΕΙ ΝΑ ΕΙΝΑΙ 4ο
#                 args.m,        # αριθμός partitions
#                 args.imbalance,
#                 True,          # suppress output
#                 args.seed,
#                 args.kahip_mode
#             )

#         except Exception as e:
#             print(">> ERROR during KaHIP execution:", e)
#             raise

#         partition = np.array(partition, dtype=np.int32)
#         print(">> KaHIP completed.")
#         print(f"   Edgecut = {edgecut}")
#         print(f"   Partition vector size = {len(partition)}")

#         # Partition statistics
#         unique, counts = np.unique(partition, return_counts=True)
#         print(">> Partition statistics:")
#         for p, c in zip(unique, counts):
#             print(f"   Part {p}: {c} nodes ({c/len(partition)*100:.2f}%)")

#         # Save partition file (for MLP training & index building)
#         part_path = os.path.join(args.index, "partition.npy")
#         os.makedirs(args.index, exist_ok=True)
#         np.save(part_path, partition)

#         print(f">> Partition saved to {part_path}")

#     except Exception as e:
#         print(f">> ERROR: {str(e)}")
#         import traceback
#         traceback.print_exc()
    
#     finally:
#         # Επαναφορά stdout
#         sys.stdout = output_logger.terminal
#         output_logger.close()
#         print(f">> All output has been saved to {output_file}")


# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
import argparse
from kahip import kaffpa
import subprocess
import numpy as np
import os
import networkx as nx
import matplotlib.pyplot as plt
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# Ανακατεύθυνση stdout σε αρχείο
class OutputLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()

# -------------------------------------------------------------
# 1. LOAD DATASET (MNIST / SIFT)
# -------------------------------------------------------------
def load_dataset(path, dtype):
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

# -------------------------------------------------------------
# 2. CALL THE C++ ivfflat_knn EXECUTABLE TO GENERATE GRAPH
# -------------------------------------------------------------
def build_knn_graph(args):
    print(">> Building k-NN graph with IVFFlat executable...")

    cmd = [
        "./../ergasia1/Project/search",
        "-ivfflat_knn", 
        "-d", args.dataset,
        "-type", args.type,
        "--knn", str(args.knn),
        "-kclusters", "14",
        "-nprobe", "2",
        "-o", "knn_output.txt"
    ]

    subprocess.run(cmd, check=True)

    if not os.path.exists("knn_output.txt"):
        raise RuntimeError("knn_output.txt was not created. Check that the executable ran correctly.")
    
    print(">> First 10 lines of knn_output.txt:")
    with open("knn_output.txt") as f:
        for i, line in enumerate(f):
            if i < 10:
                print(f"Line {i}: {line.strip()}")
            else:
                break 
    print(">> Finished running IVFFlat, reading knn_output.txt")
    return read_knn_graph("knn_output.txt")

def read_knn_graph(filename):
    graph = {}
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            node, neighs = line.split(":")
            node = int(node)
            neigh_list = neighs.strip().split()
            graph[node] = [int(n) for n in neigh_list]
    return graph

# -------------------------------------------------------------
# 3. MAKE GRAPH UNDIRECTED + ADD WEIGHTS (1 or 2)
# -------------------------------------------------------------
def make_weighted_undirected(graph):
    """Δημιουργία συμμετρικού γράφου με ΣΩΣΤΑ βάρη"""
    # Πρώτα, αφαίρεση self-loops και διπλοεγγραφών
    cleaned_graph = {}
    for u in graph:
        # Αφαίρεση self-loops και διπλοεγγραφών, διατήρηση μόνο μοναδικών γειτόνων
        unique_neighbors = list(set([nb for nb in graph[u] if nb != u]))
        cleaned_graph[u] = unique_neighbors
    
    # Υπολογισμός βαρών
    edge_weights = {}
    for u in cleaned_graph:
        for v in cleaned_graph[u]:
            edge = tuple(sorted((u, v)))
            edge_weights[edge] = edge_weights.get(edge, 0) + 1
    
    # Δημιουργία τελικού συμμετρικού γράφου
    adj = {}
    for (u, v), count in edge_weights.items():
        weight = 2 if count == 2 else 1
        
        if u not in adj:
            adj[u] = []
        if v not in adj:
            adj[v] = []
            
        # Προσθήκη και από τις δύο πλευρές με το ΙΔΙΟ βάρος
        adj[u].append((v, weight))
        adj[v].append((u, weight))
    
    return adj

def check_graph_consistency(adj):
    """Έλεγχος ότι ο γράφος είναι συμμετρικός"""
    print(">> Checking graph consistency...")
    
    inconsistencies = 0
    for u in adj:
        for (v, w1) in adj[u]:
            # Βρες το αντίστροφο edge
            found = False
            reverse_weight = None
            if v in adj:
                for (u2, w2) in adj[v]:
                    if u2 == u:
                        found = True
                        reverse_weight = w2
                        break
            
            if not found:
                print(f"ERROR: Edge {u}→{v} exists but {v}→{u} missing!")
                inconsistencies += 1
            elif w1 != reverse_weight:
                print(f"ERROR: Edge {u}→{v} has weight {w1} but {v}→{u} has weight {reverse_weight}")
                inconsistencies += 1
    
    if inconsistencies == 0:
        print("✓ Graph is perfectly symmetric!")
    else:
        print(f"✗ Found {inconsistencies} inconsistencies")
    
    return inconsistencies == 0

def visualize_graph(adj, max_nodes=20):
    """Οπτικοποίηση γράφου (προαιρετικό)"""
    if len(adj) > max_nodes:
        print(f">> Graph too large for visualization (> {max_nodes} nodes)")
        return
        
    G = nx.Graph()
    for u, neighbors in adj.items():
        for v, w in neighbors:
            G.add_edge(u, v, weight=w)
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=300, node_color="skyblue")
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.savefig('graph_visualization.png')
    print(">> Graph visualization saved as 'graph_visualization.png'")
    plt.close()

# 4. CONVERT TO CSR FOR KAHIP
def to_csr(adj, n):
    print(">> Converting graph to CSR format...")
    xadj = [0]
    adjncy = []
    adjcwgt = []
    vwgt = [1] * n  # all vertex weights = 1

    for i in range(n):
        neighbors = adj.get(i, [])
        for (v, w) in neighbors:
            adjncy.append(v)
            adjcwgt.append(w)
        xadj.append(len(adjncy))

    return vwgt, xadj, adjncy, adjcwgt

# -------------------------------------------------------------
# 5. MLP MODEL WITH SOFTMAX
# -------------------------------------------------------------
class MLPClassifier(nn.Module):
    def __init__(self, d_in, n_out, hidden=64, layers=3):
        super().__init__()
        layers_list = []
        dim = d_in
        
        # Input layer
        layers_list.append(nn.Linear(d_in, hidden))
        layers_list.append(nn.ReLU())
        
        # Hidden layers
        for _ in range(layers - 1):
            layers_list.append(nn.Linear(hidden, hidden))
            layers_list.append(nn.ReLU())
        
        # Output layer (χωρίς activation - θα προστεθεί softmax στο forward)
        layers_list.append(nn.Linear(hidden, n_out))
        
        self.net = nn.Sequential(*layers_list)

    def forward(self, x):
        logits = self.net(x)
        return logits  # Επιστροφή logits για CrossEntropyLoss

def train_model(model, train_loader, val_loader, epochs, optimizer, criterion, device):
    """Εκπαίδευση μοντέλου με validation"""
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            train_total += yb.size(0)
            train_correct += (predicted == yb).sum().item()
        
        train_accuracy = 100 * train_correct / train_total
        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(train_accuracy)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item()
                
                _, predicted = torch.max(logits, 1)
                val_total += yb.size(0)
                val_correct += (predicted == yb).sum().item()
        
        val_accuracy = 100 * val_correct / val_total
        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(val_accuracy)
        
        print(f'Epoch {epoch+1}/{epochs}: '
              f'Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracy:.2f}%, '
              f'Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accuracy:.2f}%')
    
    return train_losses, val_losses, train_accuracies, val_accuracies

# -------------------------------------------------------------
# MAIN FUNCTION
# -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", dest="dataset", required=True)
    parser.add_argument("-i", dest="index", required=True)
    parser.add_argument("-type", dest="type", required=True)

    parser.add_argument("--knn", type=int, default=10)
    parser.add_argument("-m", type=int, default=100)
    parser.add_argument("--imbalance", type=float, default=0.03)
    parser.add_argument("--kahip_mode", type=int, default=2)

    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--nodes", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)

    parser.add_argument("--seed", type=int, default=1)

    args = parser.parse_args()

    # Ορισμός seed για reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Ανακατεύθυνση output σε αρχείο
    output_file = "nlsh_build_output.txt"
    output_logger = OutputLogger(output_file)
    sys.stdout = output_logger
    
    try:
        print("=" * 60)
        print("NEURAL LSH BUILD PROCESS STARTED")
        print("=" * 60)
        
        # STEP 1: LOAD DATASET
        print(f">> Loading dataset: {args.dataset}")
        data = load_dataset(args.dataset, args.type)
        n = data.shape[0]
        d = data.shape[1]
        print(f">> Dataset loaded: {n} vectors, {d} dimensions")

        # STEP 2: BUILD kNN GRAPH USING C++
        knn_graph = build_knn_graph(args)
        print(f">> k-NN graph built with {len(knn_graph)} nodes")

        # STEP 3: PREPARE GRAPH FOR KAHIP 
        print(">> Making graph undirected and weighted...")
        adj = make_weighted_undirected(knn_graph)

        check_graph_consistency(adj)
        
        print(">> Graph structure summary:")
        print(f"   Total nodes: {len(adj)}")
        total_edges = sum(len(neighbors) for neighbors in adj.values()) // 2
        print(f"   Total edges: {total_edges}")
        
        # Αποθήκευση γραφικής αναπαράστασης (μόνο για μικρούς γράφους)
        visualize_graph(adj)

        # Χρησιμοποιούμε τον πραγματικό αριθμό κόμβων του γράφου
        n_nodes = len(adj)
        vwgt, xadj, adjncy, adjcwgt = to_csr(adj, n_nodes)

        # ---------------------------------------------------------
        # STEP 4: RUN KAHIP PARTITIONING
        # ---------------------------------------------------------
        print(">> Running KaHIP...")

        try:
            edgecut, partition = kaffpa(
                vwgt,          # vertex weights
                xadj,          # xadj array
                adjncy,        # adjacency list
                adjcwgt,       # edge weights  
                args.m,        # αριθμός partitions
                args.imbalance,
                True,          # suppress output
                args.seed,
                args.kahip_mode
            )

        except Exception as e:
            print(">> ERROR during KaHIP execution:", e)
            raise

        partition = np.array(partition, dtype=np.int32)
        print(">> KaHIP completed.")
        print(f"   Edgecut = {edgecut}")
        print(f"   Partition vector size = {len(partition)}")

        # Partition statistics
        unique, counts = np.unique(partition, return_counts=True)
        print(">> Partition statistics:")
        for p, c in zip(unique, counts):
            print(f"   Part {p}: {c} nodes ({c/len(partition)*100:.2f}%)")

        # ---------------------------------------------------------
        # STEP 5: TRAIN THE MLP CLASSIFIER (ΔΙΟΡΘΩΜΕΝΟ)
        # ---------------------------------------------------------
        print(">> Preparing data for MLP training...")
        
        # ΔΙΟΡΘΩΣΗ: Δημιουργία labels για ΟΛΑ τα σημεία του dataset
        # Υποθέτουμε ότι οι κόμβοι του γράφου αντιστοιχούν 1-1 με τα σημεία του dataset
        if n_nodes != n:
            print(f"WARNING: Graph has {n_nodes} nodes but dataset has {n} points")
            print(">> Using first {n_nodes} points for training")
            # Χρησιμοποιούμε μόνο τα σημεία που έχουν partition labels
            X_data = data[:n_nodes]
            y_labels = partition
        else:
            X_data = data
            y_labels = partition
        
        print(f">> Final training data: {X_data.shape[0]} samples, {args.m} classes")
        
        # Δημιουργία TensorDataset
        X_tensor = torch.tensor(X_data, dtype=torch.float32)
        y_tensor = torch.tensor(y_labels, dtype=torch.long)
        
        # Split σε train/validation (80/20)
        dataset_size = len(X_tensor)
        indices = list(range(dataset_size))
        split = int(0.8 * dataset_size)
        
        train_indices = indices[:split]
        val_indices = indices[split:]
        
        train_dataset = torch.utils.data.Subset(TensorDataset(X_tensor, y_tensor), train_indices)
        val_dataset = torch.utils.data.Subset(TensorDataset(X_tensor, y_tensor), val_indices)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        
        # Ορισμός συσκευής
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f">> Using device: {device}")
        
        # Δημιουργία μοντέλου
        model = MLPClassifier(
            d_in=X_data.shape[1], 
            n_out=args.m,
            hidden=args.nodes, 
            layers=args.layers
        ).to(device)
        
        print(f">> MLP Model: {sum(p.numel() for p in model.parameters())} parameters")
        
        # Ορισμός optimizer και loss function
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss()
        
        # Εκπαίδευση
        print(">> Starting MLP training...")
        train_losses, val_losses, train_accuracies, val_accuracies = train_model(
            model, train_loader, val_loader, args.epochs, optimizer, criterion, device
        )
        
        print(">> Training completed.")
        
        # Αποθήκευση μοντέλου
        model_path = os.path.join(args.index, "model.pth")
        os.makedirs(args.index, exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'input_dim': X_data.shape[1],
            'output_dim': args.m,
            'hidden_dim': args.nodes,
            'num_layers': args.layers
        }, model_path)
        print(">> Saved model to", model_path)
        
        # Αποθήκευση partition
        part_path = os.path.join(args.index, "partition.npy")
        np.save(part_path, y_labels)
        print(f">> Partition saved to {part_path}")
        
        # Αποθήκευση inverted file
        print(">> Building inverted file...")
        inverted_file = {}
        for point_idx, part_label in enumerate(y_labels):
            if part_label not in inverted_file:
                inverted_file[part_label] = []
            inverted_file[part_label].append(point_idx)
        
        inv_file_path = os.path.join(args.index, "inverted_file.npy")
        np.save(inv_file_path, inverted_file)
        print(f">> Inverted file saved to {inv_file_path}")
        
        print("=" * 60)
        print("NEURAL LSH BUILD PROCESS COMPLETED SUCCESSFULLY")
        print("=" * 60)

    except Exception as e:
        print(f">> ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Επαναφορά stdout
        sys.stdout = output_logger.terminal
        output_logger.close()
        print(f">> All output has been saved to {output_file}")


if __name__ == "__main__":
    main()