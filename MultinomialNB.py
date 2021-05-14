#  Copyright (c) 2021 Jäger. All rights reserved.

import numpy as np
from math import log, pow, exp


class MultinomialNB:
    def __init__(self) -> None:
        self.y_labels = []
        self.grouped_Xs = []
        self.class_priors = []
        self.likelihoods_by_class = []
        pass

    def fit(self, vec_X, y, idf=None):
        self.y_labels = np.unique(y).tolist()
        self.grouped_Xs = [vec_X[y == self.y_labels[i]] for i in range(len(self.y_labels))]
        self.class_priors = [vec_x.shape[0] / vec_X.shape[0] for vec_x in self.grouped_Xs]
        if idf is None:
            self.likelihoods_by_class = [(vec_x.sum(axis=0) + 1) / (vec_x.sum() + vec_X.shape[1]) for vec_x in
                                         self.grouped_Xs]
        else:
            self.likelihoods_by_class = [(vec_x.sum(axis=0) * idf + 1) / (sum(vec_x.sum(axis=0) * idf) + vec_X.shape[1])
                                         for vec_x in self.grouped_Xs]

    def predict(self, vectorized_samples):
        predictions = []
        for sample in vectorized_samples:
            temp = []  # list of 4 posterior_pr
            for c in range(len(self.grouped_Xs)):
                sample_likelihood = 1.0
                # log_sample_likelihood = log(sample_likelihood)
                for i in range(len(sample)):  # for word in words
                    for j in range(sample[i]):
                        # log_sample_likelihood = log_sample_likelihood + log(self.likelihoods_by_class[c][i])
                        sample_likelihood = sample_likelihood * self.likelihoods_by_class[c][i]
                # temp.append(exp(sample_likelihood + log(self.class_priors[c])))
                temp.append(sample_likelihood*self.class_priors[c])
            # print(temp)
            predictions.append(self.y_labels[temp.index(np.max(temp))])
        return predictions

def get_accuracy(estimates, true_val, verbose = False):
    n = 0
    for pair in np.array([estimates, true_val]).transpose():
        if pair[0] == pair[1]:
            n += 1
    if verbose:
        print(f"Correct: {n}\tAccuracy: {n / len(estimates):.4%}")
    return n / len(estimates)


def cross_val_score(X: np.ndarray, y: np.ndarray, idf=None, cv: int = 5):
    index_list = np.arange(X.shape[0])
    indices_folds = np.asarray(np.array_split(index_list, cv))
    results = np.zeros(cv)
    for i in range(cv):
        val_fold_indices = indices_folds[i]
        # np.delete(indices_folds, i)
        train_fold_indices = indices_folds[[j for j in range(cv) if j!=i]].flatten()
        X_val = X[val_fold_indices]
        y_val = y[val_fold_indices]
        X_train = X[train_fold_indices]
        y_train = y[train_fold_indices]
        model = MultinomialNB()
        if idf is not None:
            model.fit(X_train, y_train, idf)
        else:
            model.fit(X_train, y_train)
        predictions = model.predict(X_val)
        results[i] = get_accuracy(predictions, y_val)
    return results
