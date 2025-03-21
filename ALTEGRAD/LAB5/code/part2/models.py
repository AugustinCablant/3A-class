"""
Deep Learning on Graphs - ALTEGRAD - Nov 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class GNN(nn.Module):
    """Simple GNN model"""
    def __init__(self, n_feat, n_hidden_1, n_hidden_2, n_class, dropout):
        super(GNN, self).__init__()

        self.fc1 = nn.Linear(n_feat, n_hidden_1)
        self.fc2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.fc3 = nn.Linear(n_hidden_2, n_class)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x_in, adj):
        ############## Tasks 10 and 13
        
        ##################
        # Layer 1
        z_0 = self.fc1(x_in)
        z_0 = self.relu(torch.mm(adj, z_0))
        z_0 = self.dropout(z_0)

        # Layer 2
        z_1 = self.fc2(z_0)
        z_1 = self.relu(torch.mm(adj, z_1))
        z_1 = self.dropout(z_1)

        # Layer 3
        x = self.fc3(z_1)

        ##################

        return F.log_softmax(x, dim=1), z_1
