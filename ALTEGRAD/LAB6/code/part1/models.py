"""
Deep Learning on Graphs - ALTEGRAD - Nov 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class GATLayer(nn.Module):
    """GAT layer"""
    def __init__(self, n_feat, n_hidden, alpha=0.05):
        super(GATLayer, self).__init__()
        self.fc = nn.Linear(n_feat, n_hidden, bias = False)
        self.a = nn.Linear(2 * n_hidden, 1)
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, x, adj):
        
        ############## Task 1
    
        ##################
        z = self.fc(x)    # linear transformation
        indices = adj.coalesce().indices()    # Get the indices of the non-zero elements of the adjacency matrix (size 2 x m)
        h = self.leakyrelu(
                            self.a(
                                torch.cat(
                                        (z[indices[0,:]], 
                                         z[indices[1,:]]), 
                                         dim=1
                                         )))
        ##################

        h = torch.exp(h.squeeze())    # apply exp
        unique = torch.unique(indices[0,:]) 
        t = torch.zeros(unique.size(0), device = x.device)
        h_sum = t.scatter_add(0, indices[0,:], h)
        h_norm = torch.gather(h_sum, 0, indices[0,:])
        alpha = torch.div(h, h_norm)
        adj_att = torch.sparse.FloatTensor(indices, alpha, torch.Size([x.size(0), x.size(0)])).to(x.device)   # attention matrix 
         
        ##################
        out = torch.sparse.mm(adj_att, z)    # apply attention matrix
        ##################

        return out, alpha


class GNN(nn.Module):
    """GNN model"""
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GNN, self).__init__()
        self.mp1 = GATLayer(nfeat, nhid)
        self.mp2 = GATLayer(nhid, nhid)
        self.fc = nn.Linear(nhid, nclass)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        

    def forward(self, x, adj):
        
        ############## Tasks 2 and 4
    
        ##################
        message_passing_layer = self.relu(self.mp1(x, adj)[0])   # first GAT layer
        dropout_layer = self.dropout(message_passing_layer)    # dropout
        message_passing_layer2, alpha = self.mp2(dropout_layer, adj)   # second GAT layer
        relu_layer2 = self.relu(message_passing_layer2)    # ReLU
        fully_connected_layer = self.fc(relu_layer2)    # fully connected layer 
        ##################

        return F.log_softmax(fully_connected_layer, dim=1), alpha
