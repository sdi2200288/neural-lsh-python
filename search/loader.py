# Βήμα: Φωρτώνει Αρχειο από το index

import os                               #βιβλιοθήκη για έλεγχο/διαχείριση αρχείων και paths 
import torch                            #βιβλιοθήκη για φόρτωση PyTorch μοντέλου 
import numpy as np                      #βιβλιοθήκη για φόρτωση numpy αρχείων 
from mlp.model import MLPClassifier     #εισαγωγή του MLP classifier που θα φορτωθεί από το checkpoint

#συνάρτηση που φορτώνει ένα αποθηκευμένο Neural LSH index
def load_index(index_path):
    #index_path/model.pth  (saved checkpoint dict)
    #index_path/inverted_file.npy  (dict: part -> list_of_point_ids)
    
    ckpt_path = os.path.join(index_path, "model.pth")           #δημιουργεί το πλήρες path προς το αρχείο με το μοντέλο
    inv_path = os.path.join(index_path, "inverted_file.npy")    #δημιουργεί το πλήρες path προς το inverted index

    if not os.path.exists(ckpt_path):   #έλεγχος ότι το αρχείο του μοντέλου υπάρχει 
        raise FileNotFoundError(f"Model checkpoint not found: {ckpt_path}")
    if not os.path.exists(inv_path):    #έλεγχος ότι το inverted file υπάρχει 
        raise FileNotFoundError(f"Inverted file not found: {inv_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu")  #φόρτωση του checkpoint, βάζει το μοντέλο στη CPU

    #δημιουργία ίδιου MLPClassifier μοντέλου με βάση τις αποθηκευμένες παραμέτρους
    model = MLPClassifier(
        d_in = checkpoint['input_dim'],             #input dimension 
        n_out = checkpoint['output_dim'],           #output dimension
        hidden = checkpoint.get('hidden_dim', 64),  #hidden layers (default 64)
        layers = checkpoint.get('num_layers', 3)    #number of layers (default 3)
    )

    model.load_state_dict(checkpoint['model_state_dict'])   #φόρτωση των weights στο μοντέλο
    model.eval()    #θέτει το μοντέλο σε evaluation mode 

    inv_file = np.load(inv_path, allow_pickle=True).item()  #φόρτωση του inverted index
    return model, inv_file      #επιστρέφει το νευρωνικό μοντέλο και το inverted index

#συνάρτηση για υπολογισμό ευκλείδειας απόστασης σε PyTorch
def euclidean_distance(query_vec, candidate_vecs):
    #query_vec: tensor [d]
    #candidate_vecs: tensor [N, d]
    #returns: tensor [N] distances

    return torch.norm(candidate_vecs - query_vec, dim=1)    #υπολογίζει και επιστρέφει το ||candidate - query||_2 για κάθε υποψήφιο σημείο
