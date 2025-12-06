# Βήμα: Εκπαίδευση Ταξινομητή (PyTorch)

import torch.nn as nn   #βιβλιοθήκη για την εισαγωγή του neural network module της PyTorch

class MLPClassifier(nn.Module): #ορισμός κλάσης ταξινομητή που κληρονομεί από nn.Module      
    def __init__(self, d_in, n_out, hidden=64, layers=3):
        super().__init__()  #αρχικοποίηση της υπερκλάσης nn.Module 
        layers_list = []    #λίστα που θα κρατήσει όλα τα layers του μοντέλου 
        dim = d_in          #αριθμός εισόδων κάθε δείγματος
        
        # Input layer
        layers_list.append(nn.Linear(d_in, hidden)) #πρώτο πλήρως συνδεδεμένο layer, από d_in σε hidden neurons
        layers_list.append(nn.ReLU())               #ενεργοποίηση του ReLU
        
        # Hidden layers
        for _ in range(layers - 1):                         #προσθέτει layers - 1 κρυφά επίπεδα  
            layers_list.append(nn.Linear(hidden, hidden))   #κάθε κρυφό επίπεδο έχει hidden -> hidden layers
            layers_list.append(nn.ReLU())                   #ReLU ενεργοποίηση 
        
        # Output layer 
        layers_list.append(nn.Linear(hidden, n_out))    #τελικό layer -> παράγει n_out logits, ένα για κάθε κατηγορία 
        
        self.net = nn.Sequential(*layers_list)  #δημιουργεί ένα συνεχόμενο νευρωνικό δίκτυο από τα layers 

    def forward(self, x):       #forward pass του μοντέλου
        logits = self.net(x)    #προώθηση των δεδομένων μέσα από το δίκτυο
        return logits           #επιστρέφει logits (χωρίς softmax) για CrossEntropyLoss
