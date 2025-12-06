#!/usr/bin/env python3
import argparse                 #βιβλιοθήκη για parsing παραμέτρων γραμμής εντολών
import numpy as np              #βιβλιοθήκη για αριθμητικές πράξεις και πίνακες 
import os                       #βιβλιοθήκη για τη διαχείριση αρχείων 
import sys                      #βιβλιοθήκη για αλληλεπίδραση με το σύστημα 
import time                     #βιβλιοθήκη για μέτρηση χρόνων 
import torch                    #βιβλιοθήκη PyTorch για μοντέλο και tensors
import torch.nn as nn           #modules νευρωνικών δικτύων  
import torch.nn.functional as F #ενεργοποιήσεις, softmax 

from utils.dataset import load_dataset                  #συνάρτηση που φορτώνει τα dataset
from search.loader import load_index                    #συνάρτηση που φορτώνει trained μοντέλο και inverted file
from search.exact import compute_true_neighbors_numpy   #συνάρτηση για exact search
from search.loader import euclidean_distance            #συνάρτηση για την ευκλείδεια απόσταση σε PyTorch


def main():
    parser = argparse.ArgumentParser(description="Neural LSH Search")   #parser CLI
    parser.add_argument("-d", dest="dataset", required=True)            #dataset path 
    parser.add_argument("-q", dest="query_file", required=True)         #queries path 
    parser.add_argument("-i", dest="index_path", required=True)         #index folder path
    parser.add_argument("-o", dest="output_file", required=True)        #output results 
    parser.add_argument("-type", dest="data_type", required=True, choices=["sift", "mnist"])    #dataset format
    parser.add_argument("-N", type=int, default=1)          #top-N neighbors
    parser.add_argument("-R", type=float, default=2000.0)   #ακτίνα για range search
    parser.add_argument("-T", type=int, default=5)          #Τ-probes (top bins)
    parser.add_argument("-range", dest="do_range_search", default="true", choices=["true", "false"])    #mode 
    args = parser.parse_args()  #διαβάζει όλα τα flags

    # load dataset & queries
    print(f">> Loading dataset: {args.dataset}")                #ενημερωτικό μήνυμα για το χρήστη
    data = load_dataset(args.dataset, args.data_type)           #φορτώνει όλα τα vectors
    n, d = data.shape                                           #n = πλήθος vectors, d = διαστάσεις 
    print(f">> Dataset loaded: {n} vectors, {d} dimensions")


    MAX_POINTS = 70000      #μέγιστο database size
    if n > MAX_POINTS:      #αν το μέγεθος από το dataset είναι μεγαλύτερο 
        print(f">> Limiting database from {n} to {MAX_POINTS} vectors")
        data = data[:MAX_POINTS]    #κόβει στα πρώτα 70.000 vectors
        n = MAX_POINTS              #ενημερώνει τη τιμή n


    print(f">> Loading queries: {args.query_file}")         #ενημερωτικό μήνυμα για το χρ΄ήστη
    queries = load_dataset(args.query_file, args.data_type) #φορτώνει query vectors 
    Q = queries.shape[0]                                    #πλήθος queries
    print(f">> Queries loaded: {Q} vectors")

    # load index
    print(">> Loading Neural LSH index...")
    model, inv_file = load_index(args.index_path)   #φορτώνει mlp και inverted file
    print(">> Inverted file loaded!")
    print(f">> Total parts (m): {len(inv_file.keys())}")    #αριθμός buckets/bins
    
    for p in sorted(inv_file.keys())[:10]:      #εμφανίζει τα πρώτα 10 bins για debugging 
        print(f"   Part {p}: {len(inv_file[p])} points")

    total_approx_time = 0.0     #συνολικός χρόνος approximate search
    total_true_time = 0.0       #συνολικός χρόνος exact search

    method_name = "Neural LSH"  #όνομα μεθόδου για output 

    results_per_query = []      #αποθήκευση metrics ανά query

    with open(args.output_file, "w") as fout:   #ανοίγει το αρχείο για να γράψει τα αποτελέσματα

        for qid, qvec in enumerate(queries):    #για κάθε query vector  
            t0_approx = time.time()             #χρονόμετρο έναρξης approximate search

            q_t = torch.tensor(qvec, dtype=torch.float32).unsqueeze(0)  #μετατροπή σε tensor [1,d]
            with torch.no_grad():       #χωρίς υπολογισμό gradients
                logits = model(q_t)     #forward pass MLP -> [1,m] logits για κάθε bin
                probs = F.softmax(logits, dim=1).squeeze(0) #softmax -> πιθανότητες για bins [m]

            T = min(args.T, probs.shape[0])             #περιορίζει Τ σε μέγιστο αριθμό bins
            top_probs, top_bins = torch.topk(probs, T)  #top-T bins με μεγαλύτερες πιθανότητες 
            top_bins = top_bins.numpy()                 #μετατροπή σε numpy array

            candidates = set()                          #αρχικοποίηση set υποψηφίων
            for b in top_bins:                          #για κάθε επιλεγμένο bin
                if b in inv_file:                       #αν υπάρχει στο inverted file
                    candidates.update(inv_file[b])      #προσθέτουμε όλα τα points αυτού του bin
            candidates = list(candidates)               #μετατροπή σε λίστα 

            approx_ids = []             #λίστα για nearest neighbors
            approx_dists = np.array([]) #αντίστοιχες αποστάσεις 

            if len(candidates) > 0: #αν υπάρχουν υποψήφιοι 
                cand_vecs = torch.tensor(data[candidates], dtype=torch.float32) #[Ncand, d]
                q_single = q_t.squeeze(0)   #query vector [d]
                dists = euclidean_distance(q_single, cand_vecs) #ευκλείδεια απόσταση [Ncand]
                
                if args.do_range_search == "true":      #αν εκτελούμε range search
                    R = args.R
                    mask = (dists <= R)                 #boolean mask για dist = R
                    mask_np = mask.numpy()              #μετατροπή σε numpy
                    final_ids = [candidates[i] for i in range(len(candidates)) if mask_np[i]]   #επιλεγμένα ids
                    final_dists = dists.numpy()[mask_np]    #αποστάσεις 
                else:   #αλλιώς top-N nearest
                    Nreq = min(args.N, len(candidates)) 
                    _, top_idx = torch.topk(-dists, Nreq)   #μικρότερες αποστάσεις
                    top_idx = top_idx.numpy()
                    final_ids = [candidates[i] for i in top_idx]
                    final_dists = dists.numpy()[top_idx]
                approx_ids = final_ids
                approx_dists = final_dists
            else:   #αν δεν υπάρχουν υποψήφιοι, κενές λίστες 
                approx_ids = []
                approx_dists = np.array([])

            t1_approx = time.time()
            t_approx = t1_approx - t0_approx
            total_approx_time += t_approx   #συσσώρευση χρόνου 

            # ---------- true exhaustive search ----------
            t0_true = time.time()
            true_topN_idx, true_topN_dists = compute_true_neighbors_numpy(qvec, data, args.N)   #πλήρης αναζήτηση
            t1_true = time.time()
            t_true = t1_true - t0_true
            total_true_time += t_true

            # ---------- metrics ----------
            true_set = set(true_topN_idx.tolist())  #set με indices πραγματικών top-N
            approx_set = set(approx_ids)            #set με indices approximate
            if args.N > 0:
                recall = len(true_set.intersection(approx_set)) / float(args.N) #fraction σωστών neighbors
            else:
                recall = 0.0

            #Average Approximation Factor (AF): we compute for each reported approx id:
            #AF_i = true_distance(approx_id) / true_distance_of_true_best_neighbor
            #then average over reported ids. If no reported ids, AF = 1.0
            if len(approx_ids) > 0:
                best_true_dist = true_topN_dists[0] if len(true_topN_dists) > 0 else 1e-12
                af_values = []
                for aid in approx_ids:
                    dist_a = np.linalg.norm(data[aid] - qvec)   #true distance του approx id
                    af_values.append(dist_a / (best_true_dist + 1e-12))
                avg_af = float(np.mean(af_values))
            else:
                avg_af = 1.0

            #QPS: number of queries processed per second using (approx + true) time per query
            total_time_for_query = t_approx + t_true
            qps = 1.0 / total_time_for_query if total_time_for_query > 0 else float("inf")

            # ---------- write output in required format ----------
            # Method name
            fout.write(f"{method_name}\n")  #γράφει το όνομα της μεθόδου 
            fout.write(f"Query: {qid}\n")   #γράφει το ID του query

            if len(approx_ids) > 0: #αν υπάρχουν nearest neighbors
                for rank, aid in enumerate(approx_ids): #για κάθε neighbor
                    fout.write(f"Nearest neighbor-{rank+1}: {aid}\n")   #γράφει το ID
                    ad = float(approx_dists[rank]) if rank < len(approx_dists) else float(np.linalg.norm(data[aid] - qvec))
                    fout.write(f"distanceApproximate: {ad:.6f}\n")  #approximate απόσταση 
                    td = float(np.linalg.norm(data[aid] - qvec))
                    fout.write(f"distanceTrue: {td:.6f}\n") #true Euclidean απόσταση
            else:   #αν δεν υπάρχουν neighbors
                fout.write(f"Nearest neighbor-1: -1\n")
                fout.write(f"distanceApproximate: {0.0:.6f}\n")
                fout.write(f"distanceTrue: {0.0:.6f}\n")

            # R-near neighbors (if range mode)
            if args.do_range_search == "true":
                fout.write("R-near neighbors:\n")
                if len(approx_ids) > 0:
                    for aid in approx_ids:
                        fout.write(f"{aid}\n")  #γράφει τα IDs που είναι εντός ακτίνας R

            # Metrics block
            fout.write(f"Average AF: {avg_af:.6f}\n")
            fout.write(f"Recall@N: {recall:.6f}\n")
            fout.write(f"QPS: {qps:.6f}\n")

            # average times up to this query:
            processed = qid + 1
            fout.write(f"tApproximateAverage: { (total_approx_time / processed):.6f}\n")    #μέσο approximate time
            fout.write(f"tTrueAverage: { (total_true_time / processed):.6f}\n")             #μέσο true time
            fout.write("\n")

            # keep results if needed later
            results_per_query.append({
                "qid": qid,
                "approx_ids": approx_ids,
                "approx_dists": approx_dists.tolist() if isinstance(approx_dists, np.ndarray) else [],
                "true_topN_idx": true_topN_idx.tolist(),
                "true_topN_dists": true_topN_dists.tolist(),
                "recall": recall,
                "avg_af": avg_af,
                "qps": qps,
                "t_approx": t_approx,
                "t_true": t_true
            })

            #εκτύπωση περιστασιακής προόδου
            if (qid + 1) % 50 == 0 or (qid + 1) == Q:
                print(f">> Processed {qid+1}/{Q} queries. Avg tApprox: {total_approx_time/(qid+1):.4f}s, Avg tTrue: {total_true_time/(qid+1):.4f}s")

    
    # -------------------------------------------
    # WRITE TOTAL METRICS (AFTER ALL QUERIES)
    # -------------------------------------------

    avg_recall_all = np.mean([r["recall"] for r in results_per_query])  #μέσο recall
    avg_af_all = np.mean([r["avg_af"] for r in results_per_query])      #μέσο AF
    avg_qps_all = np.mean([r["qps"] for r in results_per_query])        #μέσο QPS
    avg_tapprox_all = total_approx_time / Q                             #μέσο approximate time
    avg_ttrue_all = total_true_time / Q                                 #μέσο true time

    with open(args.output_file, "a") as fout:
        fout.write("===== TOTAL RESULTS =====\n")
        fout.write(f"Total Queries: {Q}\n")
        fout.write(f"Database Size: {n}\n")
        fout.write(f"Average AF (all queries): {avg_af_all:.6f}\n")
        fout.write(f"Average Recall@{args.N}: {avg_recall_all:.6f}\n")
        fout.write(f"Average QPS: {avg_qps_all:.6f}\n")
        fout.write(f"Average tApproximate: {avg_tapprox_all:.6f}\n")
        fout.write(f"Average tTrue: {avg_ttrue_all:.6f}\n")
        fout.write("=========================\n\n")

    print(">> Search completed successfully.")
    print(f">> Results written to: {args.output_file}")

if __name__ == "__main__":  #εκκίνηση της main() συνάρτησης
    main()

