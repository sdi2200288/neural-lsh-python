import numpy as np      #βιβλιοθήκη για αριθμητικές πράξεις και χειρισμό πινάκων 
import os               #βιβλιοθήκη για λειτουργίες συστήματος αρχείων και φακέλων

#load dataset (mnist or sift)
def load_dataset(path, dtype):
    #debug μηνύματα για το path, την ύπαρξη του αρχειου και το current directory     
    print(f">> DEBUG: Loading dataset from '{path}'")
    print(f">> DEBUG: File exists: {os.path.exists(path)}")
    print(f">> DEBUG: Current directory: {os.getcwd()}")

    print(f">> Attempting to load dataset from: {path}")
    print(f">> File exists: {os.path.exists(path)}")
    
    if not os.path.exists(path):    #αν το αχείο δεν υπάρχει στο path 
        possible_paths = [          #δοκιμάζει να το βρει από μια λίστα με εναλλακτικά path
            path,
            f"../{path}",
            f"../../{path}",
            f"sift_data/{os.path.basename(path)}",
            f"../sift_data/{os.path.basename(path)}"
        ]
        for p in possible_paths:    #ψάχνει σε κάθε πιθανό path  
            if os.path.exists(p):   #αν το βρει ενήμερώνει με μήνυμα και χρησιμοποιεί αυτό το path 
                print(f">> DEBUG: Found file at: {p}")
                path = p
                break
        else:                       #αν δε βρεθεί, εμφανίζει μήνυμα σφάλματος 
            raise FileNotFoundError(f"Dataset file not found: {path}")
    
    if dtype == "sift":     #αν είναι το αρχείο τύπου sift κα΄λεί την αντίστοιχη συνάρτηση για να το διαβάσει 
        return load_fvecs(path)
    elif dtype == "mnist":  #αν είναι το αρχείο τύπου mnist κα΄λεί την αντίστοιχη συνάρτηση για να το διαβάσει 
        return load_mnist(path)
    else:                   #αν είναι άλλου τύπου ενημερώνει με μήνυμα σφάλματος 
        raise ValueError(f"Unknown dataset type: {dtype}")

#φορτώνει το sift dataset από το δυαδικό αρχείο
def load_fvecs(path):
    with open(path, "rb") as f:                             #ανοίγουμε το αρχείο σε binary mode 
        dims = np.fromfile(f, dtype=np.int32, count=1)[0]   #διαβάζουμε την πρώτη τιμή, dimension κάθε vector 
    
    data = np.fromfile(path, dtype=np.int32)                #φορτώνουμε όλο το αρχείο σαν int32
    d = data[0]                                             #το πρώτο στοιχείο είναι το dimension
    assert d == dims                                        #κάνουμε επιβεβαίωση ότι δεν έχει κάποιο πρόβλημα το αρχείο 

    #κάθε εγγραφή έχει ένα int32 που δείχνει dim + dim float32 τιμές 
    data = data.reshape(-1, d + 1)                          #ομαδοποιεί τα δεδομένα σε blocks (1 + dim)
    return data[:, 1:].view(np.float32)                     #επιστρέφει μόνο τα float32 vectors, χωρίς τη τιμή dim 

#φορτώνει το mnist idx3 αρχείο
def load_mnist(path):
    with open(path, "rb") as f:                             #ανοίγουμε το mnist αρχείο
        magic = np.frombuffer(f.read(4), dtype=">i4")[0]    #διαβάζουμε το μαγικό αριθμό του αρχείου, ο οποίος είναι 2051 για το mnist
        if magic != 2051:                                   #αν δε διαβάσουμε το 2051, τότε εμφανίζει μήνυμα σφάλματος
            raise ValueError(f"Invalid MNIST image file magic number: {magic}")

        num_images = np.frombuffer(f.read(4), dtype=">i4")[0]   #κρατάμε πόσες εικόνες έχει το αρχείο 
        rows = np.frombuffer(f.read(4), dtype=">i4")[0]         #κρατάμε το πλήθος των γραμμών κάθε εικόνας 
        cols = np.frombuffer(f.read(4), dtype=">i4")[0]         #κρατάμε το πλήθος των στηλών κάθε εικόνας 

        data = np.frombuffer(f.read(), dtype=np.uint8)          #διαβάζουμς όλα τα pixel data των εικόνων 

    data = data.reshape(num_images, rows * cols)                #μετασχηματίζουμε κάθε εικόνα σε ένα vector γραμμής 
    return data.astype(np.float32) / 255.0                      #κάνουμε κανονικοποίηση σε [0,1] και επιστρέφουμε το αποτέλεσμα 
