"""
Learning on Sets and Graph Generative Models - ALTEGRAD - Nov 2024
"""

import numpy as np


def create_train_dataset():
    n_train = 10 ** 4
    max_train_card = 10

    ############## Task 1
    
    ##################
    X_train = list()
    y_train = list()
    for _ in range(n_train):
        M = np.random.randint(1, max_train_card + 1)
        sample = np.random.randint(1, 11, M)
        sample = np.pad(sample, pad_width = (10-sample.shape[0], 0), constant_values = 0 )
        X_train.append(sample)
        y_train.append(np.sum(sample))
    ##################

    return X_train, y_train


def create_test_dataset():
    n_test = 20 ** 4 
    ############## Task 2
    
    ##################
    X_test = list()
    y_test = list()
    
    cards = range(5, 101, 5)
    n_sample_per_card = n_test // len(cards)
    for card in cards:
        for _ in range(n_sample_per_card):
            sample = np.random.randint(1, 11, card)
            sample = np.pad(sample, pad_width = (100 - sample.shape[0], 0), constant_values = 0 )
            X_test.append(sample)
            y_test.append(np.sum(sample))
    ##################

    return X_test, y_test
