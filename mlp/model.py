# Βήμα: Εκπαίδευση Ταξινομητή (PyTorch)

import torch.nn as nn
# -------------------------------------------------------------
# 5. MLP MODEL WITH SOFTMAX
# -------------------------------------------------------------
class MLPClassifier(nn.Module):
    def __init__(self, d_in, n_out, hidden=64, layers=3):
        super().__init__()
        layers_list = []
        dim = d_in
        
        # Input layer
        layers_list.append(nn.Linear(d_in, hidden))
        layers_list.append(nn.ReLU())
        
        # Hidden layers
        for _ in range(layers - 1):
            layers_list.append(nn.Linear(hidden, hidden))
            layers_list.append(nn.ReLU())
        
        # Output layer (χωρίς activation - θα προστεθεί softmax στο forward)
        layers_list.append(nn.Linear(hidden, n_out))
        
        self.net = nn.Sequential(*layers_list)

    def forward(self, x):
        logits = self.net(x)
        return logits  # Επιστροφή logits για CrossEntropyLoss
