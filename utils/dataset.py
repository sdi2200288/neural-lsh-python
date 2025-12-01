import numpy as np
import os

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