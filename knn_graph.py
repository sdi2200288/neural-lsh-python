import random
import math
import numpy as np
from typing import List

def Initialize(self, dataset_n: int) -> None:
    """
    Συνάρτηση που κάνει αρχικοποίηση τυχαίων διανυσμάτων (v) κ μετατοπίσεων (t)
    """
    # print("Initialize random components")
    
    # ρύθμιση γεννήτριων τυχαίων αριθμών 
    # Στην Python χρησιμοποιούμε το built-in random και numpy.random
    # normal_dist = normal(0.0, 1.0) - κανονική κατανομή για διανύσματα
    # uniform_dist = uniform(0.0, w) - ομοιόμορφη για μετατοπίσεις
    
    # αρχικοποίηση δομών για τυχαίες μεταβλητές
    # L πίνακες × k διανύσματα
    self.randomvectors = [[] for _ in range(self.L)]
    self.randomshifts = [[] for _ in range(self.L)]
    self.random_r = [[] for _ in range(self.L)]

    for i in range(self.L):
        self.randomvectors[i] = [[] for _ in range(self.k)]
        self.randomshifts[i] = [0.0] * self.k
        self.random_r[i] = [0] * self.k

        for j in range(self.k):
            # δημιουργία τυχαίου διανύσματος από κανονική κατανομή
            self.randomvectors[i][j] = np.random.normal(0.0, 1.0, self.dimension).tolist()
            
            # τυχαία μετατόπιση - ομοιόμορφη κατανομή
            self.randomshifts[i][j] = random.uniform(0.0, self.w)
            
            # τυχαίος συντελεστής r - ομοιόμορφη κατανομή
            self.random_r[i][j] = random.randint(1, self.M - 1)

    if dataset_n > 0:  # υπολογισμός μεγέθους πίνακα κατακερματισμού
        self.tableSize = max(1, dataset_n // 4)  # 25% του μεγέθους dataset
    
    # print("Random components initialized successfully")

def euclidean_distance(self, v1: List[float], v2: List[float]) -> float:
    """
    Συνάρτηση που υπολογίζει την ευκλείδεια απόσταση μεταξύ 2 διανυσμάτων
    """
    sum_sq = 0.0

    for i in range(len(v1)):
        diff = v1[i] - v2[i]
        sum_sq += diff * diff
    
    return math.sqrt(sum_sq)


def ANN(self, query: List[float], dataset: List[List[float]]) -> List[Tuple[int, float]]:
    """
    Συνάρτηση που κάνει προσεγγιστική αναζήτηση πλησιέστερων γειτόνων
    """
    candidate_set = set()  # σύνολο για αποφυγή διπλότυπων

    for i in range(self.L):  # συλλογή υποψήφιων από όλους τους πίνακες κατακερματισμού
        query_hash = self.CreateHFun(query, i)  # υπολογισμός hash για το query

        if query_hash in self.hashTables[i]:  # έλεγχος αν το bucket υπάρχει στον τρέχοντα πίνακα
            bucket_points = self.hashTables[i][query_hash]
            candidate_set.update(bucket_points)  # προσθήκη όλων των σημείων από το bucket στο candidate set

    candidate_distances = []  # υπολογισμός αποστάσεων για όλους τους υποψήφιους
    for j in candidate_set:
        dist = self.euclidean_distance(query, dataset[j])
        candidate_distances.append((j, dist))

    # ταξινόμηση κατά αύξουσα απόσταση
    candidate_distances.sort(key=lambda x: x[1])

    if len(candidate_distances) > self.N:  # περικοπή στους N πλησιέστερους
        candidate_distances = candidate_distances[:self.N]

    return candidate_distances
