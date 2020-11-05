from __future__ import print_function, division
import numpy as np
from enum import IntEnum

from sklearn.base import BaseEstimator, ClassifierMixin

from EKNN.utils import anydistance
from EKNN.supervised_learning import eknn_table


class kln_density(BaseEstimator, ClassifierMixin):
    """ K Nearest Neighbors classifier.

    Parameters:
    -----------
    k: int
        The number of closest neighbors that will determine the class of the
        sample that we wish to predict.
    """

    def __init__(self, e=5,metric='correlation'):

        self.e = e
        self.metric=metric
        class statType(IntEnum):
            meanofavg = 0
            stdofavg = 1
            meanofmax = 2
            stdofmax = 3

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
        self.EKNN_Table_, self.EKNN_Distances_  = ek.build_table()
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
                mask = self.y_train_ == c
                idx = np.argpartition([anydistance(test_sample, x,self.metric) for x in self.X_train_[mask]],self.e)[:self.e]
                for j in idx:
                    cat_scores[c] += self.compute_score(test_sample, idx,mask)
                # Label sample as the most common class label
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
                mask = self.y_train_ == c
                idx = np.argpartition([anydistance(test_sample, x,self.metric) for x in self.X_train_[mask]],self.e)[:self.e]
                cat_scores[c] += self.compute_score(test_sample, idx,mask)
                # Label sample as the most common class label
            tot_score= np.sum(cat_scores)
            if tot_score > 0.0:
                y_pred[i] = cat_scores/tot_score
            else:
                for c in all_labels:
                    # Sort the training samples by their distance to the test sample and get the K nearest
                    mask = self.y_train_ == c
                    idx = range(len(self.X_train_[mask]))
                    cat_scores[c] += self.compute_score(test_sample, idx, mask)
                tot_score = np.sum(cat_scores)
                if tot_score > 0.0:
                    y_pred[i] = cat_scores / tot_score
                else:
                    y_pred[i] = cat_scores
        return y_pred

    def compute_score(self, test_sample, idx,mask):
        idx_score = 0
        mylst=np.array(self.EKNN_Distances_)[mask]
        for  x in idx:
            for  z in mylst[x]:  # values not keys
                if anydistance(test_sample, self.X_train_[mask][x],self.metric) <= z:
                    idx_score += 1.0
                else:
                    s = z / anydistance(test_sample,self.X_train_[mask][x],self.metric)
                    idx_score += s if s > 0.90 else 0.0

        return idx_score


