# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 10:16:00 2017

@author: bryacha
"""

import json
import pandas as pd 
import numpy as np 
np.random.seed(1337)

#import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda#, Activation#, Input
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D#, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
#from keras.utils import np_utils
from sklearn.ensemble import RandomForestClassifier


INPUT_PATH='C:\\Users\\bryacha\\projects\\Kaggle\\Iceberg\\data\\processed\\' 

def get_model_only_HH_CNN():
    ## Build model
    model = Sequential()
    #model.add(BatchNormalization(axis=-1, input_shape=(75,75,3)))
    model.add(Conv2D(16, (5,5), input_shape=(75,75,1), activation="relu", padding="valid"))
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
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation="sigmoid"))
    optimizer = Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.1)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

def get_model_all_CNN():
    ## Build model
    model = Sequential()

    model.add(Lambda(lambda x: x, input_shape=(75, 75, 3)))
    
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(32, (3,3), activation="relu", padding="valid"))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(2,2))
#    model.add(Dropout(0.2))
    
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(64, (3,3), activation="relu", padding="valid"))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(2,2))
#    model.add(Dropout(0.2))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3,3), activation="relu", padding="valid"))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(2,2))
#    model.add(Dropout(0.2))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3,3), activation="relu", padding="valid"))
    model.add(BatchNormalization(axis=-1))
#    model.add(MaxPooling2D(2,2))
#    model.add(Dropout(0.2))
    
    model.add(Flatten())

    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(1, activation="sigmoid"))
    optimizer = Adam(lr=0.0001)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

def get_model_Comb_CNN():
    ## Build model
    ## Build model
    model = Sequential()
    #model.add(BatchNormalization(axis=-1, input_shape=(75,75,3)))
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
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation="sigmoid"))
    optimizer = Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.1)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

def get_model_Comb_CNN2():
    ## Build model
    ## Build model
    model = Sequential()

    model.add(Lambda(lambda x: x, input_shape=(75, 75, 2)))
    
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(32, (3,3), activation="relu", padding="valid"))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(2,2))
    model.add(Dropout(0.2))
    
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(64, (3,3), activation="relu", padding="valid"))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(2,2))
    model.add(Dropout(0.2))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3,3), activation="relu", padding="valid"))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(2,2))
    model.add(Dropout(0.2))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3,3), activation="relu", padding="valid"))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(2,2))
    model.add(Dropout(0.2))
    
    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))

    model.add(Dense(1, activation="sigmoid"))
    optimizer = Adam(lr=0.0001)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

def get_scaled_imgs(df):
    imgs = []
    for i, row in df.iterrows():
        #make 75x75 image
        band_1 = np.array(row['band_1']).reshape(75, 75)
        band_2 = np.array(row['band_2']).reshape(75, 75)
        band_3 = band_1 + band_2 # plus since log(x*y) = log(x) + log(y)
        
        # Rescale
        a = (band_1 - band_1.mean()) / (band_1.max() - band_1.min())
        b = (band_2 - band_2.mean()) / (band_2.max() - band_2.min())
        c = (band_3 - band_3.mean()) / (band_3.max() - band_3.min())

        imgs.append(np.dstack((a, b, c)))

    return np.array(imgs)

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
Ytrain = np.array(df_train['is_iceberg'])
Xtrain = get_scaled_imgs(df_train) 

batch_size = 32

### ----

#model = get_model_only_HH_CNN()
#model.summary()
#
#earlyStopping = EarlyStopping(monitor='val_loss', patience=25, verbose=0, mode='min')
#mcp_save = ModelCheckpoint('.mdl_wtsH.hdf5', save_best_only=True, monitor='val_loss', mode='min')
#
#model.fit(XtrainHH, Ytrain, batch_size=batch_size, epochs=100, verbose=1, callbacks=[earlyStopping, mcp_save], validation_split=0.2)
#
#model.load_weights(filepath = '.mdl_wtsH.hdf5')
#
#score = model.evaluate(XtrainHH, Ytrain, verbose=2)
#print('Train score:', score[0])
#print('Train accuracy:', score[1])
#
#predHH_test = model.predict(XtestHH, verbose=1, batch_size=256)
#print(predHH_test.reshape(predHH_test.shape[0]))
#
#predHH_train = model.predict(XtrainHH, verbose=1, batch_size=256)
#print(predHH_train.reshape(predHH_train.shape[0]))

### ----
#
#model = get_model_Comb_CNN2()
#model.summary()
#
#earlyStopping = EarlyStopping(monitor='val_loss', patience=25, verbose=0, mode='min')
#mcp_save = ModelCheckpoint('.mdl_wtsC.hdf5', save_best_only=True, monitor='val_loss', mode='min')
#tensorboard = TensorBoard(log_dir='../logs', histogram_freq=0, write_graph=True, write_images=False)
#model.fit(XtrainComb, Ytrain, batch_size=batch_size, epochs=100, verbose=1, callbacks=[earlyStopping, mcp_save,tensorboard], validation_split=0.25)
#
#model.load_weights(filepath = '.mdl_wtsC.hdf5')
#
#score = model.evaluate(XtrainComb, Ytrain, verbose=2)
#print('Train score:', score[0])
#print('Train accuracy:', score[1])
#
#predC_test = model.predict(XtestComb, verbose=1, batch_size=256)
#print(predC_test.reshape(predC_test.shape[0]))
#
#predC_train = model.predict(XtrainComb, verbose=1, batch_size=256)
#print(predC_train.reshape(predC_train.shape[0]))


model = get_model_all_CNN()
model.summary()

earlyStopping = EarlyStopping(monitor='val_loss', patience=25, verbose=0, mode='min')
mcp_save = ModelCheckpoint('.mdl_wtsA.hdf5', save_best_only=True, monitor='val_loss', mode='min')
tensorboard = TensorBoard(log_dir='../logs', histogram_freq=0, write_graph=True, write_images=False)
model.fit(Xtrain, Ytrain, batch_size=batch_size, epochs=100, verbose=1, callbacks=[earlyStopping, mcp_save,tensorboard], validation_split=0.25, shuffle=False)

model.load_weights(filepath = '.mdl_wtsA.hdf5')

score = model.evaluate(Xtrain, Ytrain, verbose=2)
print('Train score:', score[0])
print('Train accuracy:', score[1])

print('Fetching Test Data...')
with open(INPUT_PATH + 'test.json') as datafile:
    data = json.load(datafile)
    
df_test = pd.DataFrame(data)
df_test.inc_angle = df_test.inc_angle.replace('na',0)
Xtest = get_scaled_imgs(df_test)

predA_train = model.predict(Xtrain)
#print(predA_train.reshape(predA_train.shape[0]))

predA_test = model.predict(Xtest)
#print(predA_test.reshape(predA_test.shape[0]))

#trRF = np.hstack((predHH_train, x_inc_ang[:,np.newaxis]))
##trRF = np.hstack((predC_train,trRF))
#testRF = np.hstack((predHH_test, t_inc_ang[:,np.newaxis]))
#testRF = np.hstack((predC_test,testRF))

#Tune the RF
#trained_RF = random_forest_classifier(trRF,Ytrain)
#print(trained_RF)
#predRF = trained_RF.predict(testRF)
##
#submission = pd.DataFrame({'id': df_test["id"], 'is_iceberg': predRF.reshape((predRF.shape[0]))})
##submission = pd.DataFrame({'id': df_test["id"], 'is_iceberg': predNN_test.reshape((predNN_test.shape[0]))})
#print(submission.head(10))

#submission.to_csv(INPUT_PATH + 'submission_HH.csv', index=False)
### ----


