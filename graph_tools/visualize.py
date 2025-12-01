import matplotlib.pyplot as plt
import networkx as nx

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