# # # kahip_interface.py
# # import kahip
# # import numpy as np

# # class KaHIPPartitioner:
# #     def __init__(self, imbalance=0.03, seed=42, mode=2):
# #         """
# #         mode: 0=FAST, 1=ECO, 2=STRONG (default)
# #         imbalance: επιτρεπόμενη ανισορροπία (default 3%)
# #         """
# #         self.imbalance = imbalance
# #         self.seed = seed
# #         self.mode = mode
    
# #     def partition_graph(self, xadj, adjncy, adjcwgt, num_parts):
# #         """
# #         Καλεί το KaHIP για partitioning
# #         """
# #         print(f"Calling KaHIP for {len(xadj)-1} nodes, {len(adjncy)} edges, {num_parts} partitions...")
        
# #         # Μετατροπή σε arrays (απαιτείται από KaHIP)
# #         xadj_array = np.array(xadj, dtype=np.int32)
# #         adjncy_array = np.array(adjncy, dtype=np.int32)
# #         adjcwgt_array = np.array(adjcwgt, dtype=np.int32)
        
# #         try:
# #             # Δοκιμή με το σωστό API του KaHIP
# #             edgecut, parts = kahip.kaffpa(
# #                 vwgt=None,  # node weights (None = uniform)
# #                 xadj=xadj_array,
# #                 adjcwgt=adjcwgt_array,
# #                 adjncy=adjncy_array,
# #                 num_parts=num_parts,
# #                 imbalance=self.imbalance,
# #                 seed=self.seed,
# #                 mode=self.mode,
# #                 suppress_output=0
# #             )
            
# #             print(f"KaHIP completed - Edgecut: {edgecut}, Partitions: {len(np.unique(parts))}")
# #             return parts
            
# #         except Exception as e:
# #             print(f"KaHIP error: {e}")
# #             print("Falling back to random partitioning...")
# #             # Επιστροφή τυχαίων partitions σε περίπτωση σφάλματος
# #             return np.random.randint(0, num_parts, size=len(xadj)-1)

# # kahip_interface.py
# import kahip
# import numpy as np

# class KaHIPPartitioner:
#     def __init__(self, imbalance=0.03, seed=42, mode=2):
#         self.imbalance = imbalance
#         self.seed = seed
#         self.mode = mode
    
#     def partition_graph(self, xadj, adjncy, adjcwgt, num_parts):
#         """
#         Καλεί το KaHIP για partitioning με το σωστό API
#         """
#         print(f"Calling KaHIP for {len(xadj)-1} nodes, {len(adjncy)} edges, {num_parts} partitions...")
        
#         # Μετατροπή σε arrays
#         xadj_array = np.array(xadj, dtype=np.int32)
#         adjncy_array = np.array(adjncy, dtype=np.int32)
#         adjcwgt_array = np.array(adjcwgt, dtype=np.int32)
        
#         try:
#             # Σωστό API based on the error message
#             edgecut, parts = kahip.kaffpa(
#                 xadj_array,      # arg0: xadj
#                 adjcwgt_array,   # arg1: adjcwgt  
#                 adjncy_array,    # arg2: adjncy
#                 None,           # arg3: vwgt (node weights)
#                 num_parts,      # arg4: num_parts
#                 self.imbalance, # arg5: imbalance
#                 False,          # arg6: suppress_output
#                 self.seed,      # arg7: seed
#                 self.mode       # arg8: mode
#             )
            
#             print(f"KaHIP completed - Edgecut: {edgecut}, Partitions: {len(np.unique(parts))}")
#             return parts
            
#         except Exception as e:
#             print(f"KaHIP error: {e}")
#             print("Falling back to K-means partitioning...")
#             return None  # Θα χρησιμοποιήσουμε K-means