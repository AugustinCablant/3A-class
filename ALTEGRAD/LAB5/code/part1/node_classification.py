"""
Deep Learning on Graphs - ALTEGRAD - Nov 2024
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import diags, eye

from sklearn.linear_model import LogisticRegression
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import accuracy_score
from deepwalk import deepwalk


# Loads the karate network
G = nx.read_weighted_edgelist('ALTEGRAD/LAB5/code/data/karate.edgelist', delimiter=' ', nodetype=int, create_using=nx.Graph())
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())

n = G.number_of_nodes()

# Loads the class labels
class_labels = np.loadtxt('ALTEGRAD/LAB5/code/data/karate_labels.txt', delimiter=',', dtype=np.int32)
idx_to_class_label = dict()
for i in range(class_labels.shape[0]):
    idx_to_class_label[class_labels[i,0]] = class_labels[i,1]

y = list()
for node in G.nodes():
    y.append(idx_to_class_label[node])

y = np.array(y)


############## Task 5

##################
nx.draw_networkx(G, node_color = y)
##################


############## Task 6
# Extracts a set of random walks from the karate network and feeds them to the Skipgram model
n_dim = 128
n_walks = 10
walk_length = 20
model = deepwalk(G, n_walks, walk_length, n_dim)    # your code here

embeddings = np.zeros((n, n_dim))
for i, node in enumerate(G.nodes()):
    embeddings[i,:] = model.wv[str(node)]

idx = np.random.RandomState(seed=42).permutation(n)
idx_train = idx[:int(0.8*n)]
idx_test = idx[int(0.8*n):]

X_train = embeddings[idx_train,:]
X_test = embeddings[idx_test,:]

y_train = y[idx_train]
y_test = y[idx_test]


############## Task 7
# Trains a logistic regression classifier and use it to make predictions


##################
LRC = LogisticRegression().fit(X_train, y_train)
y_pred = LRC.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("The accuracy of the DeepWalk technique is: ", acc)
##################


############## Task 8
# Generates spectral embeddings

##################
A = nx.adjacency_matrix(G)
inverse_degree_sequence = [1 / G.degree(node) for node in G.nodes()]
D = diags(inverse_degree_sequence)
laplacian = eye(G.number_of_nodes()) - D @ A

_, eigenvectors = eigs(laplacian, k=2, which='SR')
eigenvectors = np.real(np.array(eigenvectors))

# Train a Logistic Regression classifier
X_train_rw = eigenvectors[idx_train,:]
X_test_rw = eigenvectors[idx_test,:]

lrc_rw = LogisticRegression().fit(X_train_rw, y_train)
y_rw_pred = lrc_rw.predict(X_test_rw)
acc_rw = accuracy_score(y_test, y_rw_pred)
print("The accuracy of the Spectral decomposition technique is: ", acc_rw)
##################
