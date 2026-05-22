# 🧠 Neural LSH — Approximate Nearest Neighbor Search with Neural Networks

> Combining k-NN graphs, graph partitioning (KaHIP), and MLP classifiers for learned vector indexing.

![Language](https://img.shields.io/badge/Language-Python%203.8%2B-blue)
![Team](https://img.shields.io/badge/Team-2%20members-green)

---

## About This Project

This is the **2nd Programming Assignment** for the course *"Software Development for Algorithmic Problems"*.

It implements **Neural LSH** — a learned approximate nearest neighbor search algorithm that replaces
classical hash functions with a trained neural network. The system builds an index over high-dimensional
vectors and supports fast multi-probe search at query time.

This assignment extends the work from [Assignment 1](../), where LSH, Hypercube, IVFFlat and IVFPQ
were implemented in C++. The same datasets (MNIST, SIFT) and evaluation metrics (QPS, Recall, AF) are used,
enabling direct performance comparison across all five algorithms.

---

## Team

| Name | Student ID |
|------|-----------|
| Παπαθανασίου Ελένη | 1115202200135 |
| Τόντου Αλτάνη-Δάφνη | 1115202200288 |

---

## How Neural LSH Works

The algorithm operates in two phases:

**Build phase** (`nlsh_build.py`):
1. Construct a **k-NN graph** from the dataset
2. Convert to an **undirected weighted graph**
3. Partition into `m` balanced parts using **KaHIP**
4. Train an **MLP classifier** to predict partition labels
5. Save index (model + inverted file structure)

**Search phase** (`nlsh_search.py`):
6. Load the saved index
7. Query using **multi-probe** technique across partitions

---

## Tech Stack

- **Python 3.8+** — main language
- **PyTorch** — MLP classifier training (GPU-accelerated if available)
- **KaHIP** — balanced graph partitioning
- **NumPy / SciPy / scikit-learn** — data processing
- **Matplotlib / Seaborn** — visualization
- **Makefile** — unified build and run system

---

## Project Structure

```
📄 nlsh_build.py          → Index construction (all training phases)
📄 nlsh_search.py         → Search using the built index

📁 knn_graph/
   ├── build_knn.py        → Build k-nearest neighbor graph
   └── read_knn.py         → Read k-NN graph from file

📁 graph_tools/
   ├── symmetric.py        → Convert to undirected weighted graph
   ├── check.py            → Graph consistency checks
   └── csr.py              → Convert to CSR format for KaHIP

📁 mlp/
   ├── model.py            → MLP classifier architecture
   └── train.py            → Training loop and utilities

📁 search/
   ├── loader.py           → Load index & compute distances
   └── exact.py            → Exact search (baseline)

📁 utils/
   ├── dataset.py          → MNIST / SIFT dataset loader
   └── logger.py           → Output logging

📄 requirements.txt
📄 Makefile
```

---

## Installation

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

Or manually:

```bash
pip install torch numpy matplotlib networkx
pip install kahip
pip install scipy scikit-learn pandas seaborn
```

---

## Build & Run

### Full pipeline

```bash
make setup          # install dependencies

make run_sift       # full Neural LSH pipeline on SIFT
make run_mnist      # full Neural LSH pipeline on MNIST
```

### Exact experiment parameters (from assignment spec)

```bash
make run_exact_sift
make run_exact_mnist
```

### Build and search separately

```bash
make build_sift     # index construction only (SIFT)
make search_sift    # search only (SIFT)

make build_mnist    # index construction only (MNIST)
make search_mnist   # search only (MNIST)
```

### Utilities

```bash
make clean_all      # remove generated files
make help           # list all available commands
```

---

### Manual usage

```bash
# Build index
python nlsh_build.py -d  -i  -type 

# Search
python nlsh_search.py -d  -q  -i  -o  -type 
```

---

## System Requirements

- **OS:** Linux
- **Python:** 3.8+
- **RAM:** 16GB+ recommended for large datasets
- **GPU:** Optional — CUDA-enabled GPU accelerates MLP training

---

## Key Concepts Demonstrated

- Graph-based indexing for Approximate Nearest Neighbor search
- Balanced graph partitioning with KaHIP
- MLP classifier training with PyTorch (supervised learning on graph structure)
- Multi-probe search strategy for improved recall
- Comparison with classical ANN methods (LSH, Hypercube, IVFFlat, IVFPQ)
- Modular Python project design with virtual environments

---

## Related

- [Assignment 1 — C++ ANN Search (LSH, Hypercube, IVFFlat, IVFPQ)](../Project)

---

*2nd Programming Assignment · Software Development for Algorithmic Problems*
