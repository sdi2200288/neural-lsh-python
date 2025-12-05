#!/usr/bin/env python3
import argparse
from kahip import kaffpa
# import subprocess
import numpy as np
import os
import networkx as nx
import matplotlib.pyplot as plt
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from knn_graph.build_knn import build_knn_graph
from knn_graph.read_knn import read_knn_graph
from graph_tools.symmetric import make_weighted_undirected
from graph_tools.check import check_graph_consistency
from graph_tools.csr import to_csr
from graph_tools.visualize import visualize_graph
from mlp.model import MLPClassifier
from mlp.train import train_model
from utils.dataset import load_dataset
from utils.logger import OutputLogger

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
    # Στο argparse του nlsh_build.py
    parser.add_argument("--knn_method", choices=["bruteforce", "ivfflat"], default="bruteforce")

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
        # knn_graph = build_knn_graph(args)
        knn_graph = build_knn_graph(args, method=args.knn_method)
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
        # visualize_graph(adj)

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
            print(f">> Using first {n_nodes} points for training")
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