#Βήμα: Κατασκευή Γράφου k-NN

#η συνάρτηση διαβάζει τον knn γράφο από το αρχείο 
def read_knn_graph(filename):
    graph = {}                  #αρχικοποιεί ένα κενό λεξικό για να αποθηκεύσει το γράφο 
    with open(filename) as f:   #ανοίγει το αρχείο για ανάγνωση
        for line in f:          #διαβάζει το αρχείο γραμμή-γραμμή 
            line = line.strip() #αφαιρεί κενά και newlines από την αρχή και το τέλος 
            if not line:        #αν η γραμμή είναι κενή, την αγνοεί 
                continue
            if ':' not in line: #αν η γραμμή δεν έχει ':', δεν είναι σωστή γραμμή γράφου και την αγνοεί
                continue
            node, neighbors = line.split(":", 1)    #χωρίζει τη γραμμή σε 2 μέρη, node και neighbors
            node = int(node.strip())                #μετατρέπει το node σε ακέραιο 
            neighbors = [int(n.strip()) for n in neighbors.strip().split()]     #για κάθε τιμή των neighbors, την μετατρέπει σε ακέραιο
            graph[node] = neighbors     #αποθηκεύει στο λεξικό τα neighbors του κάθε node 
    return graph    #επιστρέφει το λεξικό που περιέχει τον knn γράφο
