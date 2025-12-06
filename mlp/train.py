# Βήμα: Εκπαίδευση Ταξινομητή (PyTorch)

import torch    #εισαγωγή της PyTorch

#συνάρτηση που κάνει εκπαίδευση μοντέλου με validation
def train_model(model, train_loader, val_loader, epochs, optimizer, criterion, device):
    train_losses = []       #λίστα για αποθήκευση του training loss ανά epoch 
    val_losses = []         #λίστα για αποθήκευση του validation loss ανά epoch
    train_accuracies = []   #λίστα για training accuracy ανά epoch  
    val_accuracies = []     #λίστα για validation accuracy ανά epoch 
    
    for epoch in range(epochs):                     #για κάθε epoch 
        # Training phase
        model.train()                               #ενεργοποιεί training mode 
        train_loss = 0.0                            #κρατάει το training loss
        train_correct = 0                           #κρατάει τι σωστές προλέψεις 
        train_total = 0                             #κρατάει το συνολικό αριθμό samples
        
        for xb, yb in train_loader:                 #για κάθε batch από το training set
            xb, yb = xb.to(device), yb.to(device)   #μεταφορά δεδομένων σε GPU/CPU
            
            optimizer.zero_grad()                   #μηδενισμός των gradients από το προηγούμενο batch 
            logits = model(xb)                      #forward pass του μοντέλου -> προβλέψεις 
            loss = criterion(logits, yb)            #υπολογισμός loss
            loss.backward()                         #υπολογισμός gradients
            optimizer.step()                        #ενημέρση των weights
            
            train_loss += loss.item()               #πρόσθεση loss του batch 
            _, predicted = torch.max(logits, 1)     #επιλογή της κλάσης με το μεγαλύτερο logit 
            train_total += yb.size(0)               #αύξηση συνολικών δειγμάτων 
            train_correct += (predicted == yb).sum().item() #καταμέτρηση σωστών προβλέψεων
        
        train_accuracy = 100 * train_correct / train_total  #υπολογισμός training accuracy
        train_losses.append(train_loss / len(train_loader)) #μέσο training loss
        train_accuracies.append(train_accuracy)             #αποθήκευση accuracy
        
        # Validation phase
        model.eval()        #ενεργοποιεί evaluation mode (χωρίς dropout)
        val_loss = 0.0      #συσσωρευτής validation loss
        val_correct = 0     #σωστές προβλέψεις στο validation set
        val_total = 0       #σύνολο validation δειγμάτων
        
        with torch.no_grad():   #απενεργοποίηση backprop για μεγαλύτερη ταχύτητα
            for xb, yb in val_loader:   #batces από το validation set
                xb, yb = xb.to(device), yb.to(device)   #μεταφορά σε GPU/CPU
                logits = model(xb)              #forward pass χωρίς gradient
                loss = criterion(logits, yb)    #validation loss
                val_loss += loss.item()         #συσσώρευση loss
                
                _, predicted = torch.max(logits, 1) #πρόβλεψη κλάσης 
                val_total += yb.size(0)             #αύξηση συνολικών δειγμάτων 
                val_correct += (predicted == yb).sum().item()   #σωστές προβλέψεις
        
        val_accuracy = 100 * val_correct / val_total    #validation accuracy
        val_losses.append(val_loss / len(val_loader))   #μέσο validation loss
        val_accuracies.append(val_accuracy)             #αποθήκευση accuracy
        
        print(f'Epoch {epoch+1}/{epochs}: '     #εκτύπωση προόδου κάθε epoch 
              f'Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracy:.2f}%, '
              f'Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accuracy:.2f}%')
    
    return train_losses, val_losses, train_accuracies, val_accuracies   #επιστρέφει τις λίστες για plotting/αξιολόγηση μετά την εκπαίδευση 