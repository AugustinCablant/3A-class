"""
Graph Mining - ALTEGRAD - Nov 2024
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


############## Task 1
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
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())

# Print the number of connected components in the graph
#  A connected component : 
# 1) Any two nodes in the subset are connected to each other by a path 
# 2) There exists no path between a node in the subset and a node not in the subset


############## Task 2
print("Number of connected components:", nx.number_connected_components(G))

if nx.is_connected(G):
    print("The graph is connected.")
else:
    print("The graph is not connected.")
    print("There are :", len(list(nx.connected_components(G))), "connected components.")
    print("The size of the largest connected component is:", len(max(nx.connected_components(G), key = len)))
    max_connected_component = max(nx.connected_components(G), key = len)
    largest_subgraph = G.subgraph(max_connected_component)
    print("Number of nodes and edges of the largest connected component are :", largest_subgraph.number_of_nodes(), "and", largest_subgraph.number_of_edges())
    print("Fraction of nodes in the largest connected component:", largest_subgraph.number_of_nodes() / G.number_of_nodes())
    print("Fraction of edges in the largest connected component:", largest_subgraph.number_of_edges() / G.number_of_edges())



