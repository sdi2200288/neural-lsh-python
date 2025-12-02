#Βήμα: Κατασκευή Γράφου k-NN

import subprocess
import os
from knn_graph.read_knn import read_knn_graph

def build_knn_graph(args, method="bruteforce"):
    """Build k-NN graph using specified method"""
    print(f">> Building k-NN graph with {method.upper()} method...")
    
    if method == "bruteforce":
        cmd = [
           "./../ergasia1/Project/search",  
            "-bruteforce_knn", 
            "-d", args.dataset,
            "-type", args.type,
            "--knn", str(args.knn),
            "-o", "knn_output.txt"
        ]
    elif method == "ivfflat":
        cmd = [
           "./../ergasia1/Project/search",
            "-ivfflat_knn", 
            "-d", args.dataset,
            "-type", args.type,
            "--knn", str(args.knn),
            "-kclusters", "100",  # παράμετροι για IVFFlat
            "-nprobe", "5", 
            "-o", "knn_output.txt"
        ]
    else:
        raise ValueError(f"Unknown method: {method}")
    
    print(f">> Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f">> Command output: {result.stdout}")
        if result.stderr:
            print(f">> Command stderr: {result.stderr}")
    except subprocess.CalledProcessError as e:
        print(f">> ERROR: Command failed with return code {e.returncode}")
        print(f">> stderr: {e.stderr}")
        raise
    
    # Έλεγχος και ανάγνωση output
    if not os.path.exists("knn_output.txt"):
        raise RuntimeError("knn_output.txt was not created")
    
    print(">> First 5 lines of knn_output.txt:")
    with open("knn_output.txt") as f:
        for i, line in enumerate(f):
            if i < 5:
                print(f"Line {i}: {line.strip()}")
            else:
                break
    
    return read_knn_graph("knn_output.txt")
