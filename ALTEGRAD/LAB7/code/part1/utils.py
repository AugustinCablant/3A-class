"""
Learning on Sets and Graph Generative Models - ALTEGRAD - Nov 2024
"""

import numpy as np


def create_train_dataset():
    n_train = 10 ** 5
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
    ############## Task 2
    
    ##################
    n_test = 10000
    step_test_card = 5
    max_test_card = 101
    min_test_card = 5
    cards = range(min_test_card, max_test_card, step_test_card)
    n_sample_per_card = n_test // len(cards)

    X_test = list()
    y_test = list()

    for card in cards:
        X = np.random.randint(1,11, size = (n_sample_per_card, card))
        y = np.sum(X, axis=1)
        X_test.append(X)
        y_test.append(y)
                      
    return X_test, y_test
