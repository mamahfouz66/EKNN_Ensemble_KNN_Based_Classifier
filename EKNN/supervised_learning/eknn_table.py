from __future__ import print_function, division
import numpy as np
from enum import IntEnum
from EKNN.utils import anydistance


class eknn_table:
    class statTyp(IntEnum):
        meanofavg = 0
        stdofavg = 1
        meanofmax = 2
        stdofmax = 3
        meanofmin = 4
        stdofmin = 5

    def __init__(self, e=5, X_train=[], y_train=[], metric='correlation'):
        self.e = e
        self.X_train = X_train
        self.y_train = y_train
        self.metric = metric

    def build_table(self):

        EKNN_Table = []
        EKNN_Distances = []
        # for x in EKNN_Table:
        #     x = dict()

        for i, test_sample in enumerate(self.X_train):
            # Sort the training samples by their distance to the test sample and get the K nearest
            X_c = self.X_train[self.y_train == self.y_train[i]]
            idx = np.argsort([anydistance(test_sample, x, self.metric) for x in X_c])[:self.e]
            # Extract the labels of the K nearest neighboring training samples
            EKNN_Table.append(idx)
            EKNN_Distances.append([anydistance(test_sample, X_c[j],self.metric) for j in idx])
            # compute statistics using EKNN_Table

        return EKNN_Table, np.array(EKNN_Distances)



    def EKNN_Statistics(self, EKNN_Distances):

        ## compute statistics  from neigbors of neigbors list nn within no. levels lvls

        all_labels = len(set(self.y_train))
        EKNN_Statistics = np.zeros(shape=(all_labels, len(self.statTyp)))
        for c in range(all_labels):

            EKNN_Statistics[c, self.statTyp.meanofavg] = np.average(np.average(EKNN_Distances[self.y_train == c],axis=1), axis=0)
            EKNN_Statistics[c, self.statTyp.stdofavg] = np.std(np.average(EKNN_Distances[self.y_train == c], axis=1),axis=0)
            EKNN_Statistics[c, self.statTyp.meanofmax] = np.average(np.max(EKNN_Distances[self.y_train == c], axis=1),axis=0)
            EKNN_Statistics[c, self.statTyp.stdofmax] = np.std(np.max(EKNN_Distances[self.y_train == c], axis=1), axis=0)
            EKNN_Statistics[c, self.statTyp.meanofmin] = np.average(np.min(EKNN_Distances[self.y_train == c], axis=1), axis=0)
            EKNN_Statistics[c, self.statTyp.stdofmin] = np.std(np.min(EKNN_Distances[self.y_train == c], axis=1), axis=0)

        return  EKNN_Statistics
