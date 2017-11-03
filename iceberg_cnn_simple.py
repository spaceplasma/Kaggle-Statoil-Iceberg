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
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping
#from keras.utils import np_utils

INPUT_PATH='C:\\Users\\bryacha\\projects\\Kaggle\\Iceberg\\data\\processed\\' 

# Read the json files into a pandas dataframe
print('Fetching Training Data...')
with open(INPUT_PATH + 'train.json') as datafile:
    data = json.load(datafile)
    
df_train = pd.DataFrame(data)

print('Fetching Test Data...')
with open(INPUT_PATH + 'test.json') as datafile:
    data = json.load(datafile)
    
df_test = pd.DataFrame(data)

Ytrain = np.array(df_train['is_iceberg'])

x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in df_train["band_1"]])
x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in df_train["band_2"]])

t_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in df_test["band_1"]])
t_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in df_test["band_2"]])


Xtrain = np.concatenate([x_band1[:, :, :, np.newaxis], x_band2[:, :, :, np.newaxis]], axis=-1)

Xtest = np.concatenate([t_band1[:, :, :, np.newaxis], t_band2[:, :, :, np.newaxis]], axis=-1)

## Build model
model = Sequential()
model.add(Conv2D(16, (3,3), input_shape=(75,75,2), data_format='channels_last', activation="elu", padding="valid"))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(1, activation="sigmoid"))
model.summary()
model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
earlyStopping = EarlyStopping(monitor='acc', patience=5, verbose=0, mode='max')
model.fit(Xtrain, Ytrain, epochs=50, verbose=1, callbacks=[earlyStopping], validation_split=0.2)

score = model.evaluate(Xtrain, Ytrain, verbose=2)
print('Train score:', score[0])
print('Train accuracy:', score[1])

pred = model.predict(Xtest, verbose=0)

submission = pd.DataFrame({'id': df_test["id"], 'is_iceberg': pred.reshape((pred.shape[0]))})
print(submission.head(10))

submission.to_csv(INPUT_PATH + 'submission.csv', index=False)

score = model.evaluate(Xtest, np.around(pred), verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

