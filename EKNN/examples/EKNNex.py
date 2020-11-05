# import seaborn as sns      #bioinformatics library
from __future__ import print_function
from sklearn.utils import class_weight
import time
from io import TextIOWrapper, BytesIO
from zipfile import ZipFile
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from scipy.io.arff import loadarff
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.datasets import make_classification
from sklearn import model_selection
from sklearn.utils import resample
# from sklearn.ensemble import StackingClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, accuracy_score, balanced_accuracy_score
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
# example of grid searching key hyperparametres for KNeighborsClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

from EKNN.supervised_learning import kln_avg, knn, kln_max,kln_min, kln_density
from EKNN.utils import  accuracy_score
from mlxtend.classifier import StackingClassifier
from sklearn.naive_bayes import GaussianNB
from pandas import read_csv, Categorical
from EKNN.supervised_learning import eknn_table

# Generate a random n-class classification problem
def get_dataset():
    ## generate datasets
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=1)
    return X, y


def read_ziparff():
    #   urllib.request.urlretrieve('https://cometa.ujaen.es/public/full/yahoo_arts.arff', 'yahoo_arts.arff')
    #   archive = ZipFile('../data/CNS.zip', 'r')
    #   in_mem_fo = TextIOWrapper(archive.open('CNS.arff'), encoding='ascii')
    #   data  = loadarff(in_mem_fo)
    data = loadarff('../data/Leukemia.arff')
    return data


def read_csvfile(dir_name, file_name,category_first=False, up_sample=False):
    # import os
    # os.chdir('/Users/stevenhurwitt/Documents/Blog/Classification')

    df = read_csv(dir_name + '/{}.csv'.format(file_name), sep=',')

    # df.info()
    if (category_first):
        # df_majority = df[df.columns[0].values == 0]
        # df_minority = df[df.balance == 1]
        # df_minority_upsampled = resample(df_minority,
        #                                  replace=True,  # sample with replacement
        #                                  n_samples=576,  # to match majority class
        #                                  random_state=123)  # reproducible results
        # Combine majority class with upsampled minority class
        # df_upsampled = pd.concat([df_majority, df_minority_upsampled])
        # Display new class counts
        # df_upsampled.balance.value_counts()
        X = np.array(df[df.columns[1:]].astype(float))  # X = df[1:,:-1]
        y = np.array(df[df.columns[0]].astype('category').cat.codes)  # y= df[1:,-1:]
    else:
        X = np.array(df[df.columns[:-1]].astype(float))  # X = df[1:,:-1]
        y = np.array(df[df.columns[-1]].astype('category').cat.codes)  # y= df[1:,-1:]

    return X, y


# get a stacking ensemble of models
def get_stacking(pip=True):
    # define the base models
    # level0 = list()
    #
    # level0.append(('knn', knn(k=3)))
    # level0.append(('kln_Avg', kln_avg(e=5)))
    # level0.append(('kln_max', kln_max(e=5)))
    # level0.append(('kln_density', kln_density(e=5)))
    # define meta learner model
    clf1 = knn(k=3)
    clf2 = kln_avg(e=5)
    clf3 = kln_max(e=5)
    clf4 = kln_density(e=5)
    clf5 = kln_min(e=5)
    lr = LogisticRegression()
    # define the stacking ensemble using sklearn.ensemble
    # model = StackingClassifier(estimators=level0, final_estimator=lr, c
    clf = StackingClassifier(classifiers=[clf1, clf2, clf3, clf4,clf5], use_probas=True, average_probas=False,
                             meta_classifier=lr)

    if pip:
        clf_steps = [('pca', PCA(n_components=28)), ('m', clf)]
        return Pipeline(steps=clf_steps)
    else:
        return clf


# get a list of models to evaluate
def get_models_pca():
    models = dict()
    knn_steps = [('pca', PCA(n_components=28)), ('m', knn(k=3))]
    models['kNN'] = Pipeline(steps=knn_steps)

    klnavg_steps = [('pca', PCA(n_components=28)), ('m', kln_avg(e=7))]
    clf1 = Pipeline(steps=klnavg_steps)
    lr = LogisticRegression()
    models['KLN-Avg'] = StackingClassifier(classifiers=[clf1], use_probas=True, average_probas=False,
                                           meta_classifier=lr)

    # clf2 = kln_max(e=5)
    klnmax_steps = [('pca', PCA(n_components=28)), ('m', kln_max(e=5))]
    models['KLN-Max'] = Pipeline(steps=klnmax_steps)

    klnmin_steps = [('pca', PCA(n_components=28)), ('m', kln_min(e=5))]
    models['KLN-Min'] = Pipeline(steps=klnmin_steps)

    # StackingClassifier(classifiers=[clf2], use_probas=True, average_probas=False,meta_classifier=lr)

    klndensity_steps = [('pca', PCA(n_components=28)), ('m', kln_density(e=5))]
    models['KDN'] = Pipeline(steps=klndensity_steps)
    models['EKNN'] = get_stacking(False)

    return models
def get_models():
# good for calculating the runtime
    models = dict()

    # models['kNN'] = knn(k=3)
    # models['KLN-Avg'] =  kln_avg(e=7)
    # models['KLN-Max'] = kln_max(e=5)
    # models['KLN-Min'] = kln_min(e=5)
    # models['KDN'] = kln_density(e=5)
    # models['EKNN'] = get_stacking(False)
    # models['NN'] = MLPClassifier()
    # models['RF'] = RandomForestClassifier()
    # models['NB'] = GaussianNB()
    # models['SVM'] = SVC()   #default rbf
    # models['kNNKD'] = KNeighborsClassifier(n_neighbors=3,algorithm='kd_tree')

    # clf5 = kln_min(e=5)
    # clf4 = kln_max(e=5)
    # lr = LogisticRegression()
    # define the stacking ensemble using sklearn.ensemble
    # model = StackingClassifier(estimators=level0, final_estimator=lr, c
    # models['KLN-Min'] = StackingClassifier(classifiers=[clf5], use_probas=True, average_probas=False, meta_classifier=lr)
    # models['KLN-Max'] = StackingClassifier(classifiers=[clf4,clf5], use_probas=True, average_probas=False, meta_classifier=lr)

    return models

def get_grids():
    grids = dict()
    grids['knn'] = dict(
        k=[1, 2, 3])  # , weights=['uniform', 'distance'],metric=['euclidean', 'manhattan', 'minkowski'])
    grids['kln_avg'] = dict(e=[3, 4, 5, 6, 7, 8, 9])
    grids['kln_max'] = dict(e=[3, 4, 5, 6, 7, 8, 9])
    grids['kln_density'] = dict(e=[3, 4, 5, 6, 7, 8, 9])
    grids['stacking'] = dict()
    return grids


# evaluate a give model using cross-validation
def evaluate_model(X, y, model, cv,scoring, params):
    class_weights = class_weight.compute_class_weight('balanced', np.unique(y), y)
    scores = model_selection.cross_validate(model, X, y, scoring= scoring,  cv=cv, n_jobs=-1, error_score='raise')
    return scores


def fine_tune(model, grid, X, y):
    scorers = {
        # 'precision_score': make_scorer(precision_score),
        # 'recall_score': make_scorer(recall_score),
        'accuracy_score': make_scorer(accuracy_score)
    }
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, error_score=0)
    clf = grid_search.fit(X, y)
    return clf


def bah_dist_Bet2NormalDist(clsdist):
    # Bhattacharyya coefficient
    up= clsdist[0]
    sp= clsdist[1]
    uq= clsdist[2]
    sq= clsdist[3]
    rat = np.power(sp, 2) /  np.power(sq, 2)
    result = 0.25 * np.log(0.25 * (rat + (1 / rat) + 2.0))
    result += 0.25 * (np.power((up - uq), 2)) / (np.power(sp, 2) + np.power(sq, 2))
    return result

def compute_bah_distance(K, X,y):
    bah_distance=dict()
    cls_distributions = dict()
    all_labels = np.unique(y)
    c1=all_labels[0]
    c2=all_labels[1]
    ek = eknn_table(e=K, X_train=X, y_train=y,metric='correlation')
    EKNN_Table , EKNN_Distances  = ek.build_table()
    EKNN_Statistics  = ek.EKNN_Statistics(EKNN_Distances)
    cls_distributions['KLN-Avg'] = [EKNN_Statistics[c1, ek.statTyp.meanofavg], EKNN_Statistics[c1, ek.statTyp.stdofavg],
                 EKNN_Statistics[c2, ek.statTyp.meanofavg],  EKNN_Statistics[c2, ek.statTyp.stdofavg]]
    bah_distance['KLN-Avg']= bah_dist_Bet2NormalDist(cls_distributions['KLN-Avg'] )
    cls_distributions['KLN-Min'] = [EKNN_Statistics[c1, ek.statTyp.meanofmin], EKNN_Statistics[c1, ek.statTyp.stdofmin],
                 EKNN_Statistics[c2, ek.statTyp.meanofmin],  EKNN_Statistics[c2, ek.statTyp.stdofmin]]
    bah_distance['KLN-Min']= bah_dist_Bet2NormalDist(cls_distributions['KLN-Min'] )
    cls_distributions['KLN-Max'] = [EKNN_Statistics[c1, ek.statTyp.meanofmax],EKNN_Statistics[c1, ek.statTyp.stdofmax],
                 EKNN_Statistics[c2, ek.statTyp.meanofmax], EKNN_Statistics[c2, ek.statTyp.stdofmax]]
    bah_distance['KLN-Max']= bah_dist_Bet2NormalDist(cls_distributions['KLN-Max'] )
    return  bah_distance, cls_distributions



def get_models_forbahat_dist(k):
    models=dict()
    lr = LogisticRegression()
    clf1 = kln_min(e=k)
    models['KLN-Min'] = StackingClassifier(classifiers=[clf1], use_probas=True, average_probas=False, meta_classifier=lr)
    clf2 = kln_max(e=k)
    models['KLN-Max'] = StackingClassifier(classifiers=[clf2], use_probas=True, average_probas=False, meta_classifier=lr)
    clf3 = kln_avg(e=k)
    models['KLN-Avg'] = StackingClassifier(classifiers=[clf3], use_probas=True, average_probas=False, meta_classifier=lr)
    clf4 =  kln_density(e=k)
    models['KDN'] = clf4
    clf5 = knn(k=k)
    models['kNN'] = clf5
    models['EKNN'] = StackingClassifier(classifiers=[clf1, clf2, clf3, clf4, clf5], use_probas=True, average_probas=False,
                        meta_classifier=lr)
    return models
def compute_bah_distance_foralldata():
    df = pd.DataFrame(columns = ['algorithm','dataset', 'K', 'bah_dist','u1', 'sgma1', 'u2','sgma2','fit_time', 'score_time', 'acc', 'balacc', 'auc'])
    #dsnames = ['KentRidge','GDS3257','nottermantrans','Leukemia', 'centralNervous','diabetes','ionosphere', 'parkinsons', 'transfusion']
    dsnames = ['ionosphere']
    ## no need for feature_selection
    dsnames = [str(x) + '_selected' for x in dsnames]
    dir_name= '../data/csv_selected'
    category_first = True
    n_splits = 5
    n_repeats = 1

    scoring = ['accuracy', 'balanced_accuracy', 'roc_auc']
    for dsname in dsnames:
        X, y = read_csvfile(dir_name, dsname, category_first)
        X = normalize(X)
        for K in range(3, 10,2):
            bahdist, clsdist  = compute_bah_distance(K,X, y)
            models = get_models_forbahat_dist(K)
            for name, model in models.items():
                p=len(df)+1
                df.loc[ p,'K'] = K
                try:
                    df.loc[p,'bah_dist'] = bahdist[name]
                    df.loc[p,'u1'] = clsdist[name][0]
                    df.loc[p, 'sgma1'] = clsdist[name][1]
                    df.loc[p,'u2'] = clsdist[name][2]
                    df.loc[p, 'sgma2'] = clsdist[name][3]
                except:
                    df.loc[p, 'bah_dist'] = '---'
                df.loc[p, 'dataset'] = dsname
                df.loc[p, 'algorithm'] = name
                cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1)
                scores = model_selection.cross_validate(model, X, y, scoring=scoring, cv=cv, n_jobs=-1, error_score='raise')
                df.loc[p, 'acc'] = (np.mean(scores['test_accuracy']))
                df.loc[p, 'balacc']= (np.mean(scores['test_balanced_accuracy']))
                df.loc[p, 'auc'] = (np.mean(scores['test_roc_auc']))

                if (name=='EKNN'):
                    df.loc[p,'fit_time'] = ((np.mean(scores['fit_time'])) / n_repeats)/3  # average fit time for one splite
                    df.loc[p,'score_time'] = ((np.mean(scores['score_time']) * n_splits) / (len(X) * n_repeats))/3
                else:
                    df.loc[p,'fit_time'] = ((np.mean(scores['fit_time'])) / n_repeats)  # average fit time for one splite
                    df.loc[p,'score_time'] = ((np.mean(scores['score_time']) * n_splits) / (len(X) * n_repeats))
        df.to_csv(r'../output/{} bah.csv'.format( dsname), sep=',')
    df = df.sort_values(by=['algorithm', 'dataset', 'K'])
    df.to_csv(r'../output/{}.csv'.format('bahdistances'), sep=',')

def main():
    # data = datasets.load_iris()
    # X = normalize(data.data)
    # y = data.target

    # or use random data
    # X, y = get_dataset()
    # get the models to evaluate

    #  X,y=read_ziparff()

    bestparams = dict()
    bestparams['kNN'] = dict(k=3)
    bestparams['KLN-Avg'] = dict(e=5)
    bestparams['KLN-Max'] = dict(e=5)
    bestparams['KLN-Min'] = dict(e=5)
    bestparams['KDN'] = dict(e=5)
    bestparams['EKNN'] = dict()
    bestparams['NN'] = dict()
    bestparams['RF'] = dict()
    bestparams['NB'] = dict()
    bestparams['SVM'] = dict()
    bestparams['kNNKD'] = dict()
    # grids = get_grids()
    # for name, model in models.items():
    #     # clf can be used directly to predict clf.predict(X_test)
    #     clf = fine_tune(model, grids[name], X, y)
    #     bestparams[name] = clf.best_params_

    # evaluate the models and store results



    df = pd.DataFrame(columns = ['algorithm', 'dataset', 'fit_time', 'score_time', 'acc', 'balacc', 'prec', 'auc'])
    results =list()
    n_splits = 5
    n_repeats = 3
    scoring = ['accuracy','balanced_accuracy', 'precision','roc_auc']
    dsnames = ['KentRidge','GDS3257','nottermantrans','Leukemia', 'centralNervous']
    feature_selection=False
    if (feature_selection):
        get_models_pca()
        dir_name = '../data/csv'
        category_first=False
    else:
        models = get_models()
        dsnames = [str(x) + '_selected' for x in dsnames]
        dir_name= '../data/csv_selected'
        category_first = True
    for name, model in models.items():
        for dsname in dsnames:
            X, y = read_csvfile(dir_name,dsname,category_first)
            X = normalize(X)
            cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1)
            # start_time = time.time()
            scores = evaluate_model(X, y, model,cv,scoring, bestparams[name])
            results.append(scores)
            #df.append(pd.Series(),ignore_index=True)
            p=len(df)+1
            df.loc[ p, 'fit_time'] = ((np.mean(scores['fit_time']))/n_repeats)   # average fit time for one splite
            df.loc[ p,'score_time'] = ((np.mean(scores['score_time']) * n_splits)/(len(X)*n_repeats))
            df.loc[p,'acc'] = (np.mean(scores['test_accuracy']))
            df.loc[p,'balacc']= (np.mean(scores['test_balanced_accuracy']))
            df.loc[p,'prec'] = (np.mean(scores['test_precision']))
            df.loc[p,'auc'] = (np.mean(scores['test_roc_auc']))
            df.loc[p,'algorithm'] = name
            df.loc[p, 'dataset'] = dsname

        #The mean score and the 95% confidence interval of the score estimate for cross_val_score are hence given by
        # print('>%s accuracy %.3f (%.3f) execution time %2.3f ms' % (name, np.mean(scores), np.std(scores), exe_time))
    # plot model performance for comparison
    df.to_csv(r'../output/{}.csv'.format('pcanewcol'), sep=',')
    # plt.boxplot(results, labels=names, showmeans=True) # results of cross_val_score
    # plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
    # plt.show()

    ##  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    ##  clf = eknn_table(e=5)
    ##  y_pred = clf.predict(X_test, X_train, y_train)
    ##accuracy = accuracy_score(y_test, y_pred)

    # print("Accuracy:", accuracy)

    # Reduce dimensions to 2d using pca and plot the results
    # Plot().plot_in_2d(X_test, y_pred, title="K Nearest Neighbors", accuracy=accuracy, legend_labels=data.target_names)


if __name__ == "__main__":
    #compute_bah_distance_foralldata()
    main()
