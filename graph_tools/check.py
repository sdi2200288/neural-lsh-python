# Βήμα: Προετοιμασία Γράφου

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