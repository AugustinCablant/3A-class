"""
Graph Mining - ALTEGRAD - Nov 2024
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import diags, eye
from random import randint
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Load the graph
path = 'ALTEGRAD/LAB4/code/datasets/CA-HepTh.txt'
G = nx.read_edgelist(
                    path,
                    comments = '#',
                    delimiter = '\t',
                    create_using = None,
                    nodetype = None,
                    data = True,
                    edgetype = None,
                    encoding = 'utf-8'
                    )


############## Task 3
# Perform spectral clustering to partition graph G into k clusters
def spectral_clustering(G, k):
    """ Perform spectral clustering to partition graph G into k clusters 
    
    Parameters
    ----------
    G = (V, E): a networkx graph
    k: number of clusters
    
    Returns
    -------
    clustering: Dictionary keyed by node to the cluster to which the node belongs

    Remark : we use the same notations as the pseudo-code in the PDF
    
    """
    ##################
    # Get the adjacency matrix A and the degree matrix D
    A = nx.adjacency_matrix(G).todense()
    degrees = dict(G.degree())
    degree_matrix = np.diag([degrees[node] for node in G.nodes()])

    # Compute the Laplacian matrix L
    L = np.eye(G.number_of_nodes()) - np.linalg.inv(degree_matrix) @ A

    # Compute the first k eigenvectors of L
    eigvals, eigvecs = eigs(L, k = k, which = 'SR')

    # Apply KMeans to the rows of the matrix of eigenvectors
    kmeans = KMeans(n_clusters = k, random_state = 0).fit(eigvecs.real)
    clustering = dict(zip(G.nodes(), kmeans.labels_))
    ##################
    return clustering

k = 10
clustering_50 = spectral_clustering(G, k)
sample_nodes = np.random.choice(list(G.nodes()), size = 500, replace=False)
H = G.subgraph(sample_nodes)
node_color = [clustering_50[node] for node in sample_nodes]

# Visualisation du sous-graphe avec les couleurs des clusters
plt.figure(figsize=(10, 10))
nx.draw(H, with_labels = True, 
        node_color = node_color, 
        cmap = plt.cm.rainbow, 
        node_size = 50, 
        font_size=10)
unique_colors = list(set(node_color))  # Liste des couleurs uniques
patches = [mpatches.Patch(color=plt.cm.rainbow(color / max(unique_colors)), 
                          label=f'Cluster {int(color)}') for color in unique_colors]
plt.legend(handles=patches, title="Clusters")

plt.title("Sample of Spectral Clustering")
plt.show()

############## Task 4

##################
# your code here #
##################




############## Task 5
# Compute modularity value from graph G based on clustering
def modularity(G, clustering):
    
    ##################
    # your code here #
    ##################
    
    
    
    
    return modularity



############## Task 6

##################
# your code here #
##################







