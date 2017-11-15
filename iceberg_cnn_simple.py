# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 10:16:00 2017

@author: bryacha
"""

import json
import pandas as pd 
import numpy as np 
#np.random.seed(666)

#import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten#, Activation, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
#from keras.utils import np_utils
from sklearn.ensemble import RandomForestClassifier


INPUT_PATH='C:\\Users\\bryacha\\projects\\Kaggle\\Iceberg\\data\\processed\\' 

def get_model_only_HH_CNN():
    ## Build model
    model = Sequential()
    #model.add(BatchNormalization(axis=-1, input_shape=(75,75,3)))
    model.add(Conv2D(16, (7,7), input_shape=(75,75,1), activation="relu", padding="valid"))
    model.add(Dropout(0.2))
    model.add(Conv2D(16, (5,5), activation="relu", padding="valid"))
    model.add(MaxPooling2D(2,2))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, (5,5), activation="relu", padding="valid"))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3,3), activation="relu", padding="valid"))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(BatchNormalization(axis=-1))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation="sigmoid"))
    optimizer = Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.1)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

#def get_model_all_CNN():
#    ## Build model
#    model = Sequential()
#    #model.add(BatchNormalization(axis=-1, input_shape=(75,75,3)))
#    model.add(Conv2D(16, (5,5), input_shape=(75,75,3), activation="relu", padding="valid"))
#    model.add(Dropout(0.2))
#    model.add(Conv2D(16, (5,5), activation="relu", padding="valid"))
#    model.add(MaxPooling2D(2,2))
#    model.add(Dropout(0.2))
#    model.add(Conv2D(32, (3,3), activation="relu", padding="valid"))
#    model.add(Dropout(0.2))
#    model.add(Conv2D(32, (3,3), activation="relu", padding="valid"))
#    model.add(Dropout(0.2))
#    model.add(Flatten())
#    model.add(BatchNormalization(axis=-1))
#    model.add(Dense(512, activation='relu'))
#    model.add(Dropout(0.3))
#    model.add(Dense(1, activation="sigmoid"))
#    optimizer = Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.1)
#    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
#
#    return model

def get_model_Comb_CNN():
    ## Build model
    model = Sequential()
    model.add(Conv2D(16, (5,5), input_shape=(75,75,2), activation="relu", padding="valid"))
    model.add(Dropout(0.2))
    model.add(Conv2D(16, (5,5), activation="relu", padding="valid"))
    model.add(MaxPooling2D(2,2))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, (3,3), activation="relu", padding="valid"))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, (3,3), activation="relu", padding="valid"))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(BatchNormalization(axis=-1))
#    model.add(Dense(512, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation="sigmoid"))
    optimizer = Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.1)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

def random_forest_classifier(features, target):
    """
    To train the random forest classifier with features and target data
    :param features:
    :param target:
    :return: trained random forest classifier
    """
    clf = RandomForestClassifier()
    clf.fit(features, target)
    return clf

# Read the json files into a pandas dataframe
print('Fetching Training Data...')
with open(INPUT_PATH + 'train.json') as datafile:
    data = json.load(datafile)
    
df_train = pd.DataFrame(data)

df_train.inc_angle = df_train.inc_angle.replace('na',0)

x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in df_train["band_1"]])
x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in df_train["band_2"]])
x_inc_ang = np.array([np.array(inc_ang).astype(np.float32) for inc_ang in df_train["inc_angle"]])

idx_x = np.where(x_inc_ang>=0)

print(np.min(x_inc_ang[idx_x[0]]),np.max(x_inc_ang[idx_x[0]]),np.mean(x_inc_ang[idx_x[0]]))

Ytrain = np.array(df_train['is_iceberg'])[idx_x[0]]

# We need to normalize the input data - feature scale each channel?
mean_band = 0.
max_band = 0.
min_band = 0.

xtrHH_band = np.zeros((len(Ytrain),75,75))
xtrHV_band = np.zeros((len(Ytrain),75,75))
xtrComb_band = np.zeros((len(Ytrain),75,75))

for i in range(0,len(Ytrain)-1):

    mean_band = np.mean(x_band1[idx_x[0][i],:,:])
    max_band = np.max(x_band1[idx_x[0][i],:,:])
    min_band = np.min(x_band1[idx_x[0][i],:,:])
    xtrHH_band[i,:,:] = (x_band1[idx_x[0][i],:,:]-mean_band)/(max_band-min_band)
 
    mean_band = np.mean(x_band2[idx_x[0][i],:,:])
    max_band = np.max(x_band2[idx_x[0][i],:,:])
    min_band = np.min(x_band2[idx_x[0][i],:,:])
    xtrHV_band[i,:,:] = (x_band2[idx_x[0][i],:,:]-mean_band)/(max_band-min_band)

    xtrComb_band[i,:,:] = (xtrHH_band[i,:,:] + xtrHV_band[i,:,:])
    mean_band = np.mean(xtrComb_band[i,:,:])
    max_band = np.max(xtrComb_band[i,:,:])
    min_band = np.min(xtrComb_band[i,:,:])
    xtrComb_band[i,:,:] = (xtrComb_band[i,:,:]-mean_band)/(max_band-min_band)

print('Fetching Test Data...')
with open(INPUT_PATH + 'test.json') as datafile:
    data = json.load(datafile)
    
df_test = pd.DataFrame(data)
##df_test=df_train

df_test.inc_angle = df_test.inc_angle.replace('na',0)

t_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in df_test["band_1"]])
t_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in df_test["band_2"]])
t_inc_ang = np.array([np.array(inc_ang).astype(np.float32) for inc_ang in df_test["inc_angle"]])

idx_t = np.where(t_inc_ang>=0)

print(np.min(t_inc_ang[idx_x[0]]),np.max(t_inc_ang[idx_x[0]]),np.mean(t_inc_ang[idx_x[0]]))

ytest = np.arange(0,np.shape(t_band1)[0])

teHH_band = np.zeros((len(ytest),75,75))
teHV_band = np.zeros((len(ytest),75,75))
teComb_band = np.zeros((len(ytest),75,75))

for i in range(0,len(ytest)-1):
    mean_band = np.mean(t_band1[idx_t[0][i],:,:])
    max_band = np.max(t_band1[idx_t[0][i],:,:])
    min_band = np.min(t_band1[idx_t[0][i],:,:])
    teHH_band[i,:,:] = (t_band1[idx_t[0][i],:,:]-mean_band)/(max_band-min_band)
 
    mean_band = np.mean(t_band2[idx_t[0][i],:,:])
    max_band = np.max(t_band2[idx_t[0][i],:,:])
    min_band = np.min(t_band2[idx_t[0][i],:,:])
    teHV_band[i,:,:] = (t_band2[idx_t[0][i],:,:]-mean_band)/(max_band-min_band)

    teComb_band[i,:,:] = (t_band1[idx_t[0][i],:,:] + t_band2[idx_t[0][i],:,:])
    mean_band = np.mean(teComb_band[i,:,:])
    max_band = np.max(teComb_band[i,:,:])
    min_band = np.min(teComb_band[i,:,:])
    teHV_band[i,:,:] = (teComb_band[i,:,:]-mean_band)/(max_band-min_band)
    
XtrainAll = np.concatenate([xtrHH_band[:, :, :, np.newaxis], xtrHV_band[:, :, :, np.newaxis], xtrComb_band[:, :, :, np.newaxis]], axis=-1)
XtrainComb = np.concatenate([xtrHV_band[:, :, :, np.newaxis], xtrComb_band[:, :, :, np.newaxis]], axis=-1)
XtrainHH = xtrHH_band[:, :, :, np.newaxis]

XtestAll = np.concatenate([teHH_band[:, :, :, np.newaxis], teHV_band[:, :, :, np.newaxis], teComb_band[:, :, :, np.newaxis]], axis=-1)
XtestComb = np.concatenate([teHV_band[:, :, :, np.newaxis], teComb_band[:, :, :, np.newaxis]], axis=-1)
XtestHH = teHH_band[:, :, :, np.newaxis]

batch_size = 64

### ----

model = get_model_only_HH_CNN()
model.summary()

earlyStopping = EarlyStopping(monitor='val_loss', patience=15, verbose=0, mode='min')
mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')

model.fit(XtrainHH, Ytrain, batch_size=batch_size, epochs=100, verbose=1, callbacks=[earlyStopping, mcp_save], validation_split=0.2)

model.load_weights(filepath = '.mdl_wts.hdf5')

score = model.evaluate(XtrainHH, Ytrain, verbose=2)
print('Train score:', score[0])
print('Train accuracy:', score[1])

predHH_test = model.predict(XtestHH, verbose=1, batch_size=200)
print(predHH_test.reshape(predHH_test.shape[0]))

predHH_train = model.predict(XtrainHH, verbose=1, batch_size=200)
print(predHH_train.reshape(predHH_train.shape[0]))

### ----

model = get_model_Comb_CNN()
model.summary()

earlyStopping = EarlyStopping(monitor='val_loss', patience=15, verbose=0, mode='min')
mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')

model.fit(XtrainComb, Ytrain, batch_size=batch_size, epochs=100, verbose=1, callbacks=[earlyStopping, mcp_save], validation_split=0.2)

model.load_weights(filepath = '.mdl_wts.hdf5')

score = model.evaluate(XtrainComb, Ytrain, verbose=2)
print('Train score:', score[0])
print('Train accuracy:', score[1])

predC_test = model.predict(XtestComb, verbose=1, batch_size=256)
print(predC_test.reshape(predC_test.shape[0]))

predC_train = model.predict(XtrainComb, verbose=1, batch_size=256)
print(predC_train.reshape(predC_train.shape[0]))


trRF = np.hstack((predC_train, x_inc_ang[:,np.newaxis]))
trRF = np.hstack((predHH_train,trRF))
testRF = np.hstack((predC_test, t_inc_ang[:,np.newaxis]))
testRF = np.hstack((predHH_test,testRF))
#
trained_RF = random_forest_classifier(trRF,Ytrain)
print(trained_RF)
predRF = trained_RF.predict(testRF)
#
submission = pd.DataFrame({'id': df_test["id"], 'is_iceberg': predRF.reshape((predRF.shape[0]))})
#submission = pd.DataFrame({'id': df_test["id"], 'is_iceberg': predNN_test.reshape((predNN_test.shape[0]))})
print(submission.head(10))

#submission.to_csv(INPUT_PATH + 'submission_HH.csv', index=False)
### ----


