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
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
#from keras.utils import np_utils

INPUT_PATH='C:\\Users\\bryacha\\projects\\Kaggle\\Iceberg\\data\\processed\\' 

def get_model_only_CNN():
    ## Build model
    model = Sequential()
    #model.add(BatchNormalization(axis=-1, input_shape=(75,75,3)))
    model.add(Conv2D(16, (7,7), input_shape=(75,75,2), activation="relu", padding="valid"))
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
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.33))
    model.add(Dense(1, activation="sigmoid"))
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    #optimizer = 'adadelta'
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


# Read the json files into a pandas dataframe
print('Fetching Training Data...')
with open(INPUT_PATH + 'train.json') as datafile:
    data = json.load(datafile)
    
df_train = pd.DataFrame(data)

print('Fetching Test Data...')
with open(INPUT_PATH + 'test.json') as datafile:
    data = json.load(datafile)
    
df_test = pd.DataFrame(data)
#df_test=df_train

Ytrain = np.array(df_train['is_iceberg'])

x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in df_train["band_1"]])
x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in df_train["band_2"]])

# We need to normalize the input data - feature scale each channel?
mean_band1 = Ytrain*0.
max_band1 = Ytrain*0.
min_band1 = Ytrain*0.
mean_band2 = Ytrain*0.
max_band2 = Ytrain*0.
min_band2 = Ytrain*0.

for i in range(0,len(x_band1)-1):
    mean_band1[i] = np.mean(x_band1[i,:,:])
    max_band1[i] = np.max(x_band1[i,:,:])
    min_band1[i] = np.min(x_band1[i,:,:])
    x_band1[i,:,:] = (x_band1[i,:,:]-mean_band1[i])/(max_band1[i]-min_band1[i])
 
    mean_band2[i] = np.mean(x_band2[i,:,:])
    max_band2[i] = np.max(x_band2[i,:,:])
    min_band2[i] = np.min(x_band2[i,:,:])
    x_band2[i,:,:] = (x_band2[i,:,:]-mean_band2[i])/(max_band2[i]-min_band2[i])

x_band3 = ((x_band1+x_band2)/2)


t_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in df_test["band_1"]])
t_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in df_test["band_2"]])

ytest = np.arange(0,np.shape(t_band1)[0])

mean_band1 = ytest*0.
max_band1 = ytest*0.
min_band1 = ytest*0.
mean_band2 = ytest*0.
max_band2 = ytest*0.
min_band2 = ytest*0.

for i in range(0,len(t_band1)-1):
    mean_band1[i] = np.mean(t_band1[i,:,:])
    max_band1[i] = np.max(t_band1[i,:,:])
    min_band1[i] = np.min(t_band1[i,:,:])
    t_band1[i,:,:] = (t_band1[i,:,:]-mean_band1[i])/(max_band1[i]-min_band1[i])
 
    mean_band2[i] = np.mean(t_band2[i,:,:])
    max_band2[i] = np.max(t_band2[i,:,:])
    min_band2[i] = np.min(t_band2[i,:,:])
    t_band2[i,:,:] = (t_band2[i,:,:]-mean_band2[i])/(max_band2[i]-min_band2[i])

t_band3 = ((t_band1+t_band2)/2)

Xtrain = np.concatenate([x_band1[:, :, :, np.newaxis], x_band2[:, :, :, np.newaxis]], axis=-1)#, x_band3[:, :, :, np.newaxis]], axis=-1)

Xtest = np.concatenate([t_band1[:, :, :, np.newaxis], t_band2[:, :, :, np.newaxis]], axis=-1)#, t_band3[:, :, :, np.newaxis]], axis=-1)

batch_size = 16

#print(df_train['band_1'][1])



### ----

model = get_model_only_CNN()
model.summary()

earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True)

model.fit(Xtrain, Ytrain, batch_size=batch_size, epochs=100, verbose=1, callbacks=[earlyStopping, mcp_save], validation_split=0.25)

model.load_weights(filepath = '.mdl_wts.hdf5')

score = model.evaluate(Xtrain, Ytrain, verbose=2)
print('Train score:', score[0])
print('Train accuracy:', score[1])

pred = model.predict(Xtest, verbose=1, batch_size=200)

submission = pd.DataFrame({'id': df_test["id"], 'is_iceberg': pred.reshape((pred.shape[0]))})
print(submission.head(10))

submission.to_csv(INPUT_PATH + 'submission.csv', index=False)
### ----


