"""
Graph Mining - ALTEGRAD - Nov 2024
"""

import networkx as nx
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx

############## Task 7
path = 'ALTEGRAD/LAB4/code/datasets/MUTAG'
#load Mutag dataset
def load_dataset():

    ##################
    # Load the dataset
    dataset = TUDataset(root = '/tmp/ENZYMES', name='ENZYMES')  # Example with ENZYMES dataset

    # Convert graphs in the dataset to NetworkX format
    Gs = [to_networkx(data, to_undirected=True) for data in dataset]

    # Extract labels for each graph in the dataset
    y = [data.y.item() for data in dataset]
    ##################

    y = [data.y.item() for data in dataset]
    return Gs, y


Gs,y = load_dataset()
G_train, G_test, y_train, y_test = train_test_split(Gs, y, test_size=0.2, random_state=42)

# Compute the shortest path kernel
def shortest_path_kernel(Gs_train, Gs_test):    
    all_paths = dict()
    sp_counts_train = dict()
    
    for i,G in enumerate(Gs_train):
        sp_lengths = dict(nx.shortest_path_length(G))
        sp_counts_train[i] = dict()
        nodes = G.nodes()
        for v1 in nodes:
            for v2 in nodes:
                if v2 in sp_lengths[v1]:
                    length = sp_lengths[v1][v2]
                    if length in sp_counts_train[i]:
                        sp_counts_train[i][length] += 1
                    else:
                        sp_counts_train[i][length] = 1

                    if length not in all_paths:
                        all_paths[length] = len(all_paths)
                        
    sp_counts_test = dict()

    for i,G in enumerate(Gs_test):
        sp_lengths = dict(nx.shortest_path_length(G))
        sp_counts_test[i] = dict()
        nodes = G.nodes()
        for v1 in nodes:
            for v2 in nodes:
                if v2 in sp_lengths[v1]:
                    length = sp_lengths[v1][v2]
                    if length in sp_counts_test[i]:
                        sp_counts_test[i][length] += 1
                    else:
                        sp_counts_test[i][length] = 1

                    if length not in all_paths:
                        all_paths[length] = len(all_paths)

    phi_train = np.zeros((len(Gs_train), len(all_paths)))
    for i in range(len(Gs_train)):
        for length in sp_counts_train[i]:
            phi_train[i,all_paths[length]] = sp_counts_train[i][length]
    
  
    phi_test = np.zeros((len(Gs_test), len(all_paths)))
    for i in range(len(Gs_test)):
        for length in sp_counts_test[i]:
            phi_test[i,all_paths[length]] = sp_counts_test[i][length]

    K_train = np.dot(phi_train, phi_train.T)
    K_test = np.dot(phi_test, phi_train.T)

    return K_train, K_test



############## Task 8
# Compute the graphlet kernel
def graphlet_kernel(Gs_train, Gs_test, n_samples=200):
    graphlets = [nx.Graph(), nx.Graph(), nx.Graph(), nx.Graph()]
    
    graphlets[0].add_nodes_from(range(3))

    graphlets[1].add_nodes_from(range(3))
    graphlets[1].add_edge(0,1)

    graphlets[2].add_nodes_from(range(3))
    graphlets[2].add_edge(0,1)
    graphlets[2].add_edge(1,2)

    graphlets[3].add_nodes_from(range(3))
    graphlets[3].add_edge(0,1)
    graphlets[3].add_edge(1,2)
    graphlets[3].add_edge(0,2)

    
    phi_train = np.zeros((len(G_train), 4))
    
    ##################
    # your code here #
    ##################

    phi_test = np.zeros((len(G_test), 4))
    
    ##################
    # your code here #
    ##################
    def generate_feature_map(graphs, graphlets, n_samples):
        """
        Generate feature maps for a list of graphs by sampling subgraphs 
        and matching them to predefined graphlets.

        Args:
            graphs (list): List of NetworkX graphs.
            graphlets (list): List of predefined NetworkX graphlets for comparison.
            n_samples (int): Number of subgraph samples per graph.

        Returns:
            np.ndarray: Feature matrix of shape (len(graphs), len(graphlets)).
        """
        # Initialize the feature matrix
        phi = np.zeros((len(graphs), len(graphlets)))

        # Helper function to match a subgraph to one of the graphlets
        def match_graphlet(subgraph):
            for idx, graphlet in enumerate(graphlets):
                if nx.is_isomorphic(subgraph, graphlet):
                    return idx
            return None

        # Iterate through the graphs and generate features
        for i, G in enumerate(graphs):
            if len(G.nodes()) < 3:  # Skip graphs with fewer than 3 nodes
                print(f"Skipping graph {i} due to insufficient nodes.")
                continue

            for _ in range(n_samples):
                # Randomly sample 3 nodes from the graph
                nodes = np.random.choice(G.nodes(), 3, replace=False)
                subgraph = G.subgraph(nodes)

                # Match the subgraph to a graphlet and increment the count
                graphlet_idx = match_graphlet(subgraph)
                if graphlet_idx is not None:
                    phi[i, graphlet_idx] += 1

        return phi

    phi_train = generate_feature_map(Gs_train, graphlets, n_samples)
    phi_test = generate_feature_map(Gs_test, graphlets, n_samples)

    K_train = np.dot(phi_train, phi_train.T)
    K_test = np.dot(phi_test, phi_train.T)

    return K_train, K_test


K_train_sp, K_test_sp = shortest_path_kernel(G_train, G_test)



############## Task 9

##################
# your code here #
##################
K_train_gl, K_test_gl = graphlet_kernel(G_train, G_test)
K_train_sp, K_test_sp = shortest_path_kernel(G_train, G_test)

# GL
print("Graphlet Kernel Matrix for Training Data (K_train_gl):")
print("")
print(K_train_gl)
print("")
print("Graphlet Kernel Matrix for Test Data (K_test_gl):")
print("")
print(K_test_gl)

print("")

# SP
print("Shortest Path Kernel Matrix for Training Data (K_train_sp):")
print("")
print(K_train_sp)
print("")
print("Shortest Path Kernel Matrix for Test Data (K_test_sp):")
print("")
print(K_test_sp)


############## Task 10

##################
# your code here #
##################
G_train, G_test, y_train, y_test = train_test_split(Gs, y, test_size=0.2, random_state=42)

K_train_sp, K_test_sp = shortest_path_kernel(G_train, G_test)
K_train_gl, K_test_gl = graphlet_kernel(G_train, G_test)

# Check the dimensions
print("K_train_sp shape:", K_train_sp.shape)
print("y_train length:", len(y_train))

# SVM training
if K_train_sp.shape[0] == len(y_train):

    clf_sp = SVC(kernel='precomputed')
    clf_sp.fit(K_train_sp, y_train)
    y_pred_sp = clf_sp.predict(K_test_sp)
    accuracy_sp = accuracy_score(y_test, y_pred_sp)
    print("Accuracy with Shortest Path Kernel:", accuracy_sp)

    clf_gl = SVC(kernel='precomputed')
    clf_gl.fit(K_train_gl, y_train)
    y_pred_gl = clf_gl.predict(K_test_gl)
    accuracy_gl = accuracy_score(y_test, y_pred_gl)
    print("Accuracy with Graphlet Kernel:", accuracy_gl)
else:
    print("Kernel matrix dimensions do not match with training labels.")