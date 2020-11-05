from __future__ import print_function, division
import numpy as np
from enum import IntEnum

from sklearn.base import BaseEstimator, ClassifierMixin

from EKNN.utils import anydistance
from EKNN.supervised_learning import eknn_table

class kln_min(BaseEstimator, ClassifierMixin):
    """ K Nearest Neighbors classifier.

    Parameters:
    -----------
    k: int
        The number of closest neighbors that will determine the class of the
        sample that we wish to predict.
    """

    def __init__(self , e=1, metric='correlation'):

        self.e = 1
        self.metric=metric



    def fit(self, X_train, y_train):
        """
        This should fit classifier. All the "work" should be done here.

        Note: assert is not a good choice here and you should rather
        use try/except blog with exceptions. This is just for short syntax.
        """

        assert (type(self.e) == int), "e parameter must be integer"
        ek = eknn_table(self.e, X_train, y_train)
        self.X_train_ = X_train
        self.y_train_ = y_train
        self.EKNN_Table_, self.EKNN_Distances_ = ek.build_table()
        self.EKNN_Statistics_ = ek.EKNN_Statistics(self.EKNN_Distances_)
        return self

    def predict(self, X_test):
        y_pred = np.empty(X_test.shape[0])
        # Determine the class of each sample
        all_labels = set(self.y_train_)
        for i, test_sample in enumerate(X_test):
            cat_scores = np.zeros(len(all_labels))
            for c in all_labels:
                # Sort the training samples by their distance to the test sample and get the K nearest
                min_dist = np.min(np.partition([anydistance(test_sample, x, self.metric) for x in self.X_train_[self.y_train_ == c]], self.e)[self.e])
                cat_scores[c] += self.compute_score(min_dist, c)
            y_pred[i] = cat_scores.argmax()
        return y_pred

    def predict_proba(self, X_test):

        # Determine the class of each sample
        all_labels = set(self.y_train_)
        y_pred = np.empty(shape=(X_test.shape[0], len(all_labels)))
        for i, test_sample in enumerate(X_test):
            cat_scores = np.zeros(len(all_labels))
            for c in all_labels:
                # Sort the training samples by their distance to the test sample and get the K nearest
                max_dist = np.min(np.partition([anydistance(test_sample, x, self.metric) for x in self.X_train_[self.y_train_ == c]], self.e)[self.e])
                cat_scores[c] += self.compute_score(max_dist, c)
            y_pred[i] = cat_scores
        return y_pred


    def compute_score(self, min_dist , c):

        return np.math.exp(-np.abs(min_dist - self.EKNN_Statistics_[c][eknn_table.statTyp.meanofavg]) / self.EKNN_Statistics_[c][eknn_table.statTyp.stdofavg])
    # def get_params(self, deep = False):
    #     return {'e': self.e}