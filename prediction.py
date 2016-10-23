import os
import csv
import math
from pykalman import KalmanFilter
import pandas as pd
import numpy as np
import scipy.linalg as la
import matplotlib.pylab as plt
from sklearn.cluster import FeatureAgglomeration
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.cross_validation import KFold
from keras.models import Model, model_from_json
from keras.utils import np_utils
from keras.layers import Input, Dense, PReLU, Dropout, Convolution1D, MaxPooling1D, LSTM, TimeDistributed, \
    Flatten, merge, BatchNormalization, MaxoutDense, GRU, Reshape, SimpleRNN, Activation
from keras.optimizers import SGD
from keras.regularizers import l2

SEED = 31
NFOLDS = 10
np.random.seed(SEED)

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
ntrain = train.shape[0]
ntest = test.shape[0]
train_test = pd.concat([train.drop('Y', axis=1), test])
train_test = train_test.drop('Time', axis=1)
movingaverage2 = pd.rolling_mean(train_test, window=2)
movingaverage3 = pd.rolling_mean(train_test, window=3)
movingaverage4 = pd.rolling_mean(train_test, window=4)
movingaverage5 = pd.rolling_mean(train_test, window=5)
train_test = pd.concat([train_test, movingaverage2, movingaverage3, movingaverage4, movingaverage5], axis=1)
print(train_test.head())
returns = np.log(train_test / train_test.shift(1)).dropna()
print(str(returns.shape))
Xtrain = np.array(returns[:ntrain - 5])
Ytrain = np.array(train['Y'])[5:]
Ytrain[Ytrain < 0] = 0
Ytrain = np_utils.to_categorical(Ytrain, nb_classes=2)
Xtest = np.array(returns[ntrain - 5:])
print(Xtrain.shape)
'''
clf = ts_classifier()
Ytrain = Ytrain.reshape((Ytrain.shape[0], 1))
Xtemp = Xtrain[-200:]
Xtrain = np.concatenate((Xtrain, Ytrain), axis=1)
clf.predict(Xtrain[:-200], Xtemp, w=4)
print(clf.performance(Xtrain[-200:, -1]))
'''

window_length = 3
Xseq = []
Yseq = []
for i in range(Xtrain.shape[0] - window_length + 1):
    Xtemp = Xtrain[i:i + window_length]
    Xseq.append(Xtemp)
    Yseq.append(Ytrain[i + window_length - 1])
Xseq = np.array(Xseq)
Yseq = np.array(Yseq)

print(Xseq.shape)
print(Yseq.shape)

# prepare test data
returns = np.array(returns)
Xtestseq = []
startindex = ntrain - 4 - window_length
for i in range(Xtest.shape[0]):
    Xtemp = returns[i + startindex:i + startindex + window_length]
    Xtestseq.append(Xtemp)
Xtestseq = np.array(Xtestseq)
print(Xtestseq.shape)


def write_to_submission(Ypred):
    with open('predictions.csv', mode='wb') as outcsv:
        writer = csv.writer(outcsv)
        header = ['Time', 'Y']
        writer.writerow(header)
        for i in range(Ypred.shape[0]):
            row = []
            row.append(3000 + i + 1)
            row.append(Ypred[i])
            writer.writerow(row)


def nnet_arch(window, nfeatures):
    inputs = Input(shape=(window, nfeatures))

    l1 = LSTM(512, return_sequences=True)(inputs)
    l1 = LSTM(512, return_sequences=True)(l1)
    l1 = TimeDistributed(Dense(1024, init='he_normal'))(l1)
    l1 = TimeDistributed(PReLU())(l1)
    l1 = TimeDistributed(Dropout(0.5))(l1)
    l1 = TimeDistributed(Dense(1024, init='he_normal'))(l1)
    l1 = TimeDistributed(PReLU())(l1)
    l1 = TimeDistributed(Dropout(0.5))(l1)
    l1 = Flatten()(l1)
    l1 = Dense(2048, init='he_normal')(l1)
    l1 = PReLU()(l1)
    l1 = Dropout(0.5)(l1)
    l1 = Dense(2048, init='he_normal')(l1)
    l1 = PReLU()(l1)
    l1 = Dropout(0.5)(l1)
    l1 = Dense(2, activation='softmax')(l1)

    model = Model(input=inputs, output=l1)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

model = nnet_arch(Xseq.shape[1], Xseq.shape[2])
model.fit(Xseq, Yseq, nb_epoch=10, batch_size=128,
          shuffle=True)

Ypred = model.predict(Xtestseq, batch_size=128)


Ypred = np.argmax(Ypred, axis=1)
Ypred[Ypred == 0] = -1
write_to_submission(Ypred)
