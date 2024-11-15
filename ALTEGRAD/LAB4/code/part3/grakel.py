import numpy as np
import re
from nltk.stem.porter import PorterStemmer
import warnings
warnings.filterwarnings('ignore')

def load_file(filename):
    labels = []
    docs =[]

    with open(filename, encoding='utf8', errors='ignore') as f:
        for line in f:
            content = line.split(':')
            labels.append(content[0])
            docs.append(content[1][:-1])
    
    return docs,labels  


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower().split()


def preprocessing(docs): 
    preprocessed_docs = []
    n_sentences = 0
    stemmer = PorterStemmer()

    for doc in docs:
        clean_doc = clean_str(doc)
        preprocessed_docs.append([stemmer.stem(w) for w in clean_doc])
    
    return preprocessed_docs
    
    
def get_vocab(train_docs, test_docs):
    vocab = dict()
    
    for doc in train_docs:
        for word in doc:
            if word not in vocab:
                vocab[word] = len(vocab)

    for doc in test_docs:
        for word in doc:
            if word not in vocab:
                vocab[word] = len(vocab)
        
    return vocab


path_to_train_set = 'ALTEGRAD/LAB4/code/datasets/train_5500_coarse.label'
path_to_test_set = 'ALTEGRAD/LAB4/code/datasets/TREC_10_coarse.label'

# Read and pre-process train data
train_data, y_train = load_file(path_to_train_set)
train_data = preprocessing(train_data)

# Read and pre-process test data
test_data, y_test = load_file(path_to_test_set)
test_data = preprocessing(test_data)

# Extract vocabulary
vocab = get_vocab(train_data, test_data)
print("Vocabulary size: ", len(vocab))


import networkx as nx
import matplotlib.pyplot as plt

# Task 11

def create_graphs_of_words(docs, vocab, window_size):
    graphs = list()
    for idx,doc in enumerate(docs):
        G = nx.Graph()
    
        ##################
        # your code here #
        ##################
        
        graphs.append(G)
    
    return graphs


# Create graph-of-words representations
G_train_nx = create_graphs_of_words(train_data, vocab, 3) 
G_test_nx = create_graphs_of_words(test_data, vocab, 3)

print("Example of graph-of-words representation of document")
nx.draw_networkx(G_train_nx[3], with_labels=True)
plt.show()


from grakel import graph_from_networkx, WeisfeilerLehman
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# Task 12

def load_mutag_dataset():
    # Load edges, graph indicators, graph labels, and node labels
    edges = np.loadtxt('ALTEGRAD/LAB4/code/datasets/MUTAG/raw/MUTAG_A.txt', delimiter=",", dtype=int)
    graph_indicator = np.loadtxt('ALTEGRAD/LAB4/code/datasets/MUTAG/raw/MUTAG_graph_indicator.txt', dtype=int)
    graph_labels = np.loadtxt('ALTEGRAD/LAB4/code/datasets/MUTAG/raw/MUTAG_graph_labels.txt', dtype=int)
    node_labels = np.loadtxt('ALTEGRAD/LAB4/code/datasets/MUTAG/raw/MUTAG_node_labels.txt', dtype=int)

    # Dictionary to hold graphs, using graph IDs
    graphs = {}
    for i, graph_id in enumerate(graph_indicator):
        if graph_id not in graphs:
            graphs[graph_id] = nx.Graph()
        graphs[graph_id].add_node(i + 1, label=node_labels[i])  # 1-based indexing with node label

    # Add edges to the corresponding graph based on node's graph ID
    for edge in edges:
        node1, node2 = edge
        graph_id = graph_indicator[node1 - 1]  # Adjust for 1-based index
        graphs[graph_id].add_edge(node1, node2)

    # Create lists of graphs and labels in the correct order
    nx_graphs = [graphs[graph_id] for graph_id in sorted(graphs.keys())]
    labels = list(graph_labels)

    return nx_graphs, labels

# Transform networkx graphs to grakel representations
nx_graphs, labels = load_mutag_dataset()
grakel_graphs = list(graph_from_networkx(nx_graphs, node_labels_tag='label'))
G_train, G_test, y_train, y_test = train_test_split(grakel_graphs, labels, test_size=0.2, random_state=42)

# Initialize a Weisfeiler-Lehman subtree kernel
gk = WeisfeilerLehman(n_iter=2, normalize=True)    # your code here #

# Construct kernel matrices
K_train = gk.fit_transform(G_train)    # your code here #
K_test = gk.transform(G_test)   # your code here #

#Task 13

# Train an SVM classifier and make predictions

##################
# your code here #
##################
clf = SVC(kernel='precomputed')
clf.fit(K_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(K_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Evaluate the predictions
print("Accuracy:", accuracy_score(y_pred, y_test))


#Task 14


##################
# your code here #
##################
from grakel import ShortestPath, RandomWalk, PyramidMatch, Propagation

kernels = {
    'Weisfeiler-Lehman': WeisfeilerLehman(n_iter=3, normalize=True),
    'Shortest Path': ShortestPath(normalize=True),
    'Random Walk': RandomWalk(normalize=True),
    'Pyramid Match': PyramidMatch(normalize=True),
    'Propagation': Propagation(normalize=True)
}

# Dictionary to store accuracy results
accuracy_results = {}

# Loop over each kernel, compute kernel matrices, and evaluate
for kernel_name, kernel in kernels.items():
    print(f"Testing kernel: {kernel_name}")

    # Compute kernel matrices
    K_train = kernel.fit_transform(G_train)  # Fit and compute kernel matrix for training
    K_test = kernel.transform(G_test)        # Compute kernel matrix for testing

    # Train SVM with precomputed kernel matrix
    clf = SVC(kernel='precomputed')
    clf.fit(K_train, y_train)

    # Predict and evaluate
    y_pred = clf.predict(K_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_results[kernel_name] = accuracy

    print(f"Accuracy with {kernel_name} kernel: {accuracy}")

# Print overall results
print("\nComparison of kernel performances:")
for kernel_name, accuracy in accuracy_results.items():
    print(f"{kernel_name} Kernel Accuracy: {accuracy}")