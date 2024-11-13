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
largest_cc = G.subgraph(max(nx.connected_components(G), key=len))

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
    inverse_degree_sequence = [1 / G.degree(node) for node in G.nodes()]
    D = diags(inverse_degree_sequence)

    # Compute the Laplacian matrix L
    L = eye(G.number_of_nodes()) - D @ A

    # Compute the first k eigenvectors of L
    eigvals, eigvecs = eigs(L, k = k, which = 'SR')

    # Apply KMeans to the rows of the matrix of eigenvectors
    U = np.real(np.array(eigvecs))
    kmeans = KMeans(n_clusters=k, n_init='auto').fit(U)
    clustering = kmeans.predict(U)
    clustering = {n: c for n, c in zip(G.nodes(), clustering)}
    ##################
    return clustering


############## Task 4

##################
k = 50
clustering_50 = spectral_clustering(largest_cc, k)
print("Number of clusters for k = 50: ", len(set(clustering_50.values())))
##################




############## Task 5
# Compute modularity value from graph G based on clustering
def modularity(G, clustering):
    """ Compute modularity value from graph G based on clustering 
    
    Parameters
    ----------
    
    G = (V, E): a networkx graph
    clustering: Dictionary keyed by node to the cluster to which the node belongs
    
    Returns
    -------
    
    modularity: The modularity value
    
    """
    ##################
    m = G.number_of_edges()
    n_c = len(set(clustering.values()))   # Number of clusters or communities 
    modularity = 0
    for index_cluster in range(n_c):
        nodes_in_cluster = [node for node, cluster in clustering.items() if cluster == index_cluster]    # Nodes in the cluster
        subgraph = G.subgraph(nodes_in_cluster)    # Subgraph induced by the nodes in the cluster
        l_c = subgraph.number_of_edges()   # Number of edges in the subgraph
        d_c = sum([G.degree(node) for node in nodes_in_cluster])   # Sum of the degrees of the nodes in the cluster
        modularity += l_c / m - (d_c / (2 * m)) ** 2
    ##################
    return modularity


############## Task 6

##################
print("Modularity value for k = 50:", modularity(largest_cc, clustering_50))
random_clustering = { node : randint(0,49) for node in largest_cc.nodes() }
random_m = modularity(largest_cc, random_clustering)
print("The modularity of the random clustering is ", random_m)

""" 
Modularity value for k = 50: 0.1966931835316136
The modularity of the random clustering is  -0.00033244187612085577
"""
##################







