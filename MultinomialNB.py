#  Copyright (c) 2021 Jäger. All rights reserved.
import numpy as np


class MultinomialNB:
    def __init__(self) -> None:
        self.y_labels = []
        self.grouped_Xs = []
        self.class_priors = []
        self.likelihoods_by_class = []
        pass

    def fit(self, vec_X, y):
        self.y_labels = np.unique(y).tolist()
        self.grouped_Xs = [vec_X[y == self.y_labels[i]] for i in range(len(self.y_labels))]
        self.class_priors = [vec_x.shape[0] / vec_X.shape[0] for vec_x in self.grouped_Xs]
        self.likelihoods_by_class = [(vec_x.sum(axis=0) + 1) / (vec_x.sum() + len(vec_X.shape[1])) for vec_x in
                                     self.grouped_Xs]

    def predict(self, vectorized_samples):
        predictions = []
        for sample in vectorized_samples:
            temp = []  # list of 4 posterior_pr
            for c in range(len(self.grouped_Xs)):
                sample_likelihood = 1.0
                for i in range(len(sample)):  # for word in words
                    for j in range(sample[i]):
                        sample_likelihood = sample_likelihood * self.likelihoods_by_class[c][i]
                temp.append(sample_likelihood * self.class_priors[c])
            print(temp)
            predictions.append(self.y_labels[temp.index(np.max(temp))])
        return predictions
