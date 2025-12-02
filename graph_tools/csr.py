# Βήμα: Προετοιμασία Γράφου

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