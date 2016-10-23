import os
import csv
import random
import matplotlib.pylab as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, MeanShift, FeatureAgglomeration, SpectralClustering, DBSCAN
from sklearn.neighbors import KNeighborsClassifier
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import cdist, pdist

SEED = 31
np.random.seed(SEED)

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
ntrain = train.shape[0]
ntest = test.shape[0]
train_test = pd.concat([train.drop('Y', axis=1), test])
train_test = train_test.drop('Time', axis=1)
print(str(train_test.shape))
returns = np.log(train_test / train_test.shift(5)).dropna()

Xtrain = returns[:ntrain - 1]
Ytrain = train['Y'][1:]
Xtest = returns[ntrain - 1:]

returns = np.array(returns)
# 0.461 , use last 1000 examples, cluster 11, feature agglomeration
t1 = returns[-1000:]
clustering_clf = FeatureAgglomeration(n_clusters=11, compute_full_tree=True)
clustering_clf.fit(t1)
labels = clustering_clf.labels_
print(labels)


with open('clusters.csv', mode='wb') as outcsv:
    writer = csv.writer(outcsv)
    header = ['Asset', 'Cluster']
    writer.writerow(header)
    for i in range(labels.shape[0]):
        row = []
        row.append('X' + str(i + 1))
        row.append(labels[i] + 1)
        writer.writerow(row)
