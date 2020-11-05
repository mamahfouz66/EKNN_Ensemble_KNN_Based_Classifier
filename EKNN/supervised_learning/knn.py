from __future__ import print_function, division
import numpy as np
from enum import IntEnum

from sklearn.base import ClassifierMixin
from sklearn.base import BaseEstimator
from EKNN.utils import  anydistance


class knn(ClassifierMixin, BaseEstimator):
    """ K Nearest Neighbors classifier.

    Parameters:
    -----------
    k: int
        The number of closest neighbors that will determine the class of the
        sample that we wish to predict.
    """

    def __init__(self, k=3, metric='correlation'):
        self.k = k
        self.metric = metric
    # def _more_tags(self):
    #     return {'requires_fit': False,
    #             'non_deterministic': True}
    def _vote(self, neighbor_labels):
        """ Return the most common class among the neighbor samples """
        counts = np.bincount(neighbor_labels.astype('int'))

        return counts.argmax()

    def _vote_score(self, neighbor_labels,all_labels ):
        """ Return the most common class among the neighbor samples """
        counts = np.bincount(neighbor_labels.astype('int'),minlength=all_labels)
        return counts/np.sum(counts)

    def fit(self, X_train, y_train):
        # self.classes_ = unique_labels(y)
        self.X_train_ = X_train
        self.y_train_ = y_train
        return self

    def predict(self, X_test):
        y_pred = np.empty(X_test.shape[0])
        # Determine the class of each sample
        for i, test_sample in enumerate(X_test):
            # Sort the training samples by their distance to the test sample and get the K nearest
            idx = np.argsort([anydistance(test_sample, x, self.metric) for x in self.X_train_])[:self.k]
            # Extract the labels of the K nearest neighboring training samples
            k_nearest_neighbors = np.array([self.y_train_[i] for i in idx])
            # Label sample as the most common class label
            y_pred[i] = self._vote(k_nearest_neighbors)
        return y_pred

    def predict_proba(self, X_test):
        all_labels = len(set(self.y_train_))
        y_pred = np.empty(shape=(X_test.shape[0], all_labels))


        # Determine the class of each sample
        for i, test_sample in enumerate(X_test):
            # Sort the training samples by their distance to the test sample and get the K nearest
            idx = np.argsort([anydistance(test_sample, x, self.metric) for x in self.X_train_])[:self.k]
            # Extract the labels of the K nearest neighboring training samples
            k_nearest_neighbors = np.array([self.y_train_[i] for i in idx])
            # Label sample as the most common class label

            y_pred[i]=self._vote_score(k_nearest_neighbors,all_labels)
        return y_pred