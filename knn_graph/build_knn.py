#Βήμα: Κατασκευή Γράφου k-NN

import subprocess       #βιβλιοθήκη για την εκτέλεση εξωτερικών προγραμμάτων
import os               #βιβλιοθήκη για λειτουργίες συστήματος αρχείων 
from knn_graph.read_knn import read_knn_graph   #συνάρτηση που διαβάζει το αρχείο knn και επιστρέφει τον γράφο 

#η συνάρτηση κατασκευάζει το knn γράφο χρησιμοποιώντας καθορισμένη μέθοδο
def build_knn_graph(args, method="bruteforce"):
    print(f">> Building k-NN graph with {method.upper()} method...")    #εκτυπώνει τη μέθοδο που θα χρησιμοποιήσει  
    
    if method == "bruteforce":  #αν η μέθοδος είναι bruteforce, κατασκευάζει την αντίστοιχη εντολή 
        cmd = [
           "./../ergasia1/Project/search",  
            "-bruteforce_knn", 
            "-d", args.dataset,
            "-type", args.type,
            "--knn", str(args.knn),
            "-o", "knn_output.txt"
        ]
    elif method == "ivfflat":   #αν η μέθοδος είναι ivfflat, κατασκευάζει την αντίστοιχη εντολή
        cmd = [
           "./../ergasia1/Project/search",
            "-ivfflat_knn", 
            "-d", args.dataset,
            "-type", args.type,
            "--knn", str(args.knn),
            "-kclusters", "100",  # παράμετροι για IVFFlat
            "-nprobe", "5", 
            "-o", "knn_output.txt"
        ]
    else:       #αν δε δωθεί γνωστή μέθοδος, εμφανίζει μήνυμα σφάλματος 
        raise ValueError(f"Unknown method: {method}")
    
    print(f">> Running: {' '.join(cmd)}")   #εκτυπώνει την εντολή σε μορφή string 
    
    try:    #εκτελεί την εντολή χρησιμοποιώντας subprocess
        result = subprocess.run(cmd,                    #η εντολή
                                check=True,             #αν αποτύχει, πηγαίνει στο exception 
                                capture_output=True,    #συλλαμβάνει stdout/stderr
                                text=True)              #επιστρέφει strings και όχι bytes
        
        print(f">> Command output: {result.stdout}")    #εμφανίζει ό,τι έγραψε το πρόγραμμα στο stdout
        if result.stderr:                               #αν υπάρχει stderr το εκτυπώνει
            print(f">> Command stderr: {result.stderr}")

    except subprocess.CalledProcessError as e:      #αν υπάρξει exception 
        print(f">> ERROR: Command failed with return code {e.returncode}")  #εμφανίζει μήνυμα σφάλματος 
        print(f">> stderr: {e.stderr}") #εμφανίζει το stderr
        raise   #ξανα πηγαίνει στο exception 
    
    if not os.path.exists("knn_output.txt"):    #έλεγχος αν δημιουργήθηκε το αρχείο εξόδου
        raise RuntimeError("knn_output.txt was not created")    #αν δεν υπάρχει, προκύπτει σφάλμα 
    
    print(">> First 5 lines of knn_output.txt:")    
    with open("knn_output.txt") as f:   #ανοίγει το αρχείο για ανάγνωση 
        for i, line in enumerate(f):    #διαβάζει γραμμή-γραμμή
            if i < 5:                   #εκτυπώνει τις 5 πρώτες γραμμές και μετά σταματά
                print(f"Line {i}: {line.strip()}")
            else:
                break
    
    return read_knn_graph("knn_output.txt")     #επιστρέφει τον knn γράφο που διάβασε από το αρχείο 
