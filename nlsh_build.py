# #!/usr/bin/env python3
# import argparse
# from kahip import kaffpa
# import subprocess
# import numpy as np
# import os
# import networkx as nx
# import matplotlib.pyplot as plt


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
#         "./../../Project/Project/search",
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

# ###def read_knn_graph(filename):
#     """Διαβάζει τον k-NN γράφο αποφεύγοντας self-loops"""
#     graph = {}
    
#     with open(filename) as f:
#         lines = f.readlines()
    
#     i = 0
#     while i < len(lines):
#         line = lines[i].strip()
        
#         if line.startswith("Query:"):
#             node_id = int(line.split()[1])
#             graph[node_id] = []
            
#             i += 1
#             while i < len(lines) and not lines[i].startswith("Query:"):
#                 neighbor_line = lines[i].strip()
#                 if neighbor_line.startswith("Nearest neighbor-"):
#                     parts = neighbor_line.split(":")
#                     if len(parts) >= 2:
#                         neighbor_id = parts[1].strip()
#                         if neighbor_id.isdigit():
#                             neighbor_id = int(neighbor_id)
#                             # ΑΠΟΦΥΓΗ SELF-LOOPS
#                             if neighbor_id != node_id:
#                                 graph[node_id].append(neighbor_id)
#                 i += 1
#         else:
#             i += 1
    
#     return graph

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
#     plt.show()

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


# # # -------------------------------------------------------------
# # # 5. TRAIN MLP CLASSIFIER
# # # -------------------------------------------------------------
# # class MLP(nn.Module):
# #     def __init__(self, dim, hidden, layers, m):
# #         super().__init__()
# #         net = []
# #         in_dim = dim
# #         for _ in range(layers):
# #             net.append(nn.Linear(in_dim, hidden))
# #             net.append(nn.ReLU())
# #             in_dim = hidden
# #         net.append(nn.Linear(hidden, m))
# #         self.net = nn.Sequential(*net)

# #     def forward(self, x):
# #         return self.net(x)


# # def train_model(data, labels, args):
# #     print(">> Training MLP classifier...")

# #     X = torch.tensor(data, dtype=torch.float32)
# #     Y = torch.tensor(labels, dtype=torch.long)

# #     model = MLP(
# #         dim=data.shape[1],
# #         hidden=args.nodes,
# #         layers=args.layers,
# #         m=args.m
# #     )

# #     optimizer = optim.Adam(model.parameters(), lr=args.lr)
# #     criterion = nn.CrossEntropyLoss()

# #     B = args.batch_size
# #     for epoch in range(args.epochs):
# #         perm = torch.randperm(len(X))
# #         X = X[perm]
# #         Y = Y[perm]

# #         for i in range(0, len(X), B):
# #             xb = X[i:i+B]
# #             yb = Y[i:i+B]

# #             pred = model(xb)
# #             loss = criterion(pred, yb)

# #             optimizer.zero_grad()
# #             loss.backward()
# #             optimizer.step()

# #         print(f"  Epoch {epoch+1}/{args.epochs}, loss={loss.item():.4f}")

# #     return model


# # # -------------------------------------------------------------
# # # 6. BUILD INVERTED FILE
# # # -------------------------------------------------------------
# # def build_inverted_file(labels):
# #     inv = {}
# #     for idx, lab in enumerate(labels):
# #         inv.setdefault(lab, []).append(idx)
# #     return inv


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

#     # STEP 1: LOAD DATASET
#     data = load_dataset(args.dataset, args.type)
#     n = data.shape[0]

#     # STEP 2: BUILD kNN GRAPH USING C++
#     knn_graph = build_knn_graph(args)

#     # STEP 3: PREPARE GRAPH FOR KAHIP 
#     adj = make_weighted_undirected(knn_graph)

#     check_graph_consistency(adj)
#     print(">> Example of k-NN graph (first 5 nodes):")
#     for i, (node, neighbors) in enumerate(adj.items()):
#         if i >= 5:  # Παρουσίαση μόνο 5 nodes για ευκρίνεια
#             break
#         neigh_str = ", ".join([f"{v}(w={w})" for v, w in neighbors[:10]])  # Πρώτοι 10 γείτονες
#         print(f"Node {node}: {neigh_str}")

#     visualize_graph(adj)

#     vwgt, xadj, adjncy, adjcwgt = to_csr(adj, n)

#     # ---------------------------------------------------------
#     # STEP 4: RUN KAHIP PARTITIONING
#     # ---------------------------------------------------------
#     print(">> Running KaHIP...")
#     partition = kaffpa(
#         n,
#         xadj,
#         adjncy,
#         vwgt,
#         adjcwgt,
#         args.m,
#         args.kahip_mode,
#         args.imbalance
#     )
#     print("Current working directory:", os.getcwd())
#     print("Trying to write to:", output_file)
#     output_file = os.path.join(os.getcwd(), "kahip_partition.txt")
#     with open(output_file, "w") as f:
#         for node_id, label in enumerate(partition):
#             f.write(f"{node_id}: {label}\n")
#     print(f">> Τα αποτελέσματα του KaHIP αποθηκεύτηκαν στο {output_file}")

#     labels = np.array(partition)

#     # # ---------------------------------------------------------
#     # # STEP 5: TRAIN MLP TO PREDICT PARTITION LABELS
#     # # ---------------------------------------------------------
#     # model = train_model(data, labels, args)

#     # # ---------------------------------------------------------
#     # # STEP 6: BUILD INVERTED FILE
#     # # ---------------------------------------------------------
#     # inverted = build_inverted_file(labels)

#     # # ---------------------------------------------------------
#     # # STEP 7: SAVE INDEX
#     # # ---------------------------------------------------------
#     # print(f">> Saving model + index to {args.index}")

#     # torch.save({
#     #     "model": model.state_dict(),
#     #     "inverted": inverted,
#     #     "labels": labels
#     # }, args.index)

#     print(">> Done.")


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
    # else:
    #     return load_mnist(path)

def load_fvecs(path):
    with open(path, "rb") as f:
        dims = np.fromfile(f, dtype=np.int32, count=1)[0]
    
    data = np.fromfile(path, dtype=np.int32)
    d = data[0]
    assert d == dims

    # Το αρχείο είναι (int32 dim + float32[d]) * N
    # Άρα κάθε vector έχει d+1 στοιχεία 4-byte
    data = data.reshape(-1, d + 1)

    # Μετατρέπουμε μόνο τα floats
    return data[:, 1:].view(np.float32)

# -------------------------------------------------------------
# 2. CALL THE C++ ivfflat_knn EXECUTABLE TO GENERATE GRAPH
# -------------------------------------------------------------
def build_knn_graph(args):
    print(">> Building k-NN graph with IVFFlat executable...")

    cmd = [
        "./../../Project/Project/search",
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

    # Ανακατεύθυνση output σε αρχείο
    output_logger = OutputLogger("output.txt")
    sys.stdout = output_logger
    
    try:
        print("=" * 60)
        print("KAHIP PARTITIONING PROCESS STARTED")
        print("=" * 60)
        
        # STEP 1: LOAD DATASET
        print(f">> Loading dataset: {args.dataset}")
        data = load_dataset(args.dataset, args.type)
        n = data.shape[0]
        print(f">> Dataset loaded: {n} vectors")

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
        
        print(">> Example of k-NN graph (first 5 nodes):")
        for i, (node, neighbors) in enumerate(adj.items()):
            if i >= 5:
                break
            neigh_str = ", ".join([f"{v}(w={w})" for v, w in neighbors[:10]])
            print(f"   Node {node}: {neigh_str}")

        # Αποθήκευση γραφικής αναπαράστασης
        visualize_graph(adj)

        vwgt, xadj, adjncy, adjcwgt = to_csr(adj, n)

        # ---------------------------------------------------------
        # STEP 4: RUN KAHIP PARTITIONING
        # ---------------------------------------------------------
        print(">> Running KaHIP partitioning...")
        print(f">> Parameters: k={args.m}, imbalance={args.imbalance}, mode={args.kahip_mode}")
        
        partition = kaffpa(
            n,
            xadj,
            adjncy,
            vwgt,
            adjcwgt,
            args.m,
            args.kahip_mode,
            args.imbalance
        )
        
        # Αποθήκευση αποτελεσμάτων
        output_file = "output.txt"
        with open(output_file, "a", encoding="utf-8") as f:
            f.write("\n" + "=" * 60 + "\n")
            f.write("KAHIP PARTITIONING RESULTS\n")
            f.write("=" * 60 + "\n")
            for node_id, label in enumerate(partition):
                f.write(f"{node_id}: {label}\n")
        
        print(f">> KaHIP partitioning completed!")
        print(f">> Results appended to {output_file}")
        
        labels = np.array(partition)
        
        # Στατιστικά partition
        unique, counts = np.unique(labels, return_counts=True)
        print(">> Partition statistics:")
        for i, (part_id, count) in enumerate(zip(unique, counts)):
            print(f"   Partition {part_id}: {count} nodes ({count/n*100:.2f}%)")
        
        print(">> Process completed successfully!")

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