# Βήμα: Προετοιμασία Γράφου

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