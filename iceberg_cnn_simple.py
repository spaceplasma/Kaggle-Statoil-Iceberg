# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 10:16:00 2017

@author: bryacha
"""

import json
import pandas as pd 
import numpy as np 
import cv2
np.random.seed(1337)

#import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, Activation#, Input
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
#from keras.utils import np_utils
#from sklearn.ensemble import RandomForestClassifier
from scipy import stats


INPUT_PATH='C:\\Users\\bryacha\\projects\\Kaggle\\Iceberg\\data\\processed\\' 

#def get_model_all_CNN(bd):
#    ## Build model
#    model = Sequential()
#
#    model.add(Lambda(lambda x: x, input_shape=(75, 75, bd)))
#    
#    model.add(ZeroPadding2D((2, 2)))
#    model.add(Conv2D(32, (5,5), activation="relu", padding="valid"))
#    model.add(BatchNormalization(axis=-1))
#    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
##    model.add(Dropout(0.2))
#    
#    model.add(ZeroPadding2D((2, 2)))
#    model.add(Conv2D(64, (3,3), activation="relu", padding="valid"))
#    model.add(BatchNormalization(axis=-1))
#    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
##    model.add(Dropout(0.2))
#
##    model.add(ZeroPadding2D((1, 1)))
#    model.add(Conv2D(128, (3,3), activation="relu", padding="valid"))
#    model.add(BatchNormalization(axis=-1))
#    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
##    model.add(Dropout(0.2))
#
#    model.add(ZeroPadding2D((1, 1)))
#    model.add(Conv2D(128, (3,3), activation="relu", padding="valid"))
#    model.add(BatchNormalization(axis=-1))
#    model.add(MaxPooling2D(2,2))
##    model.add(Dropout(0.2))
#
#    model.add(ZeroPadding2D((1, 1)))
#    model.add(Conv2D(2, (3, 3), activation='relu'))
#    model.add(GlobalAveragePooling2D())
#    
##    model.add(Flatten())
#
##    model.add(Dense(64, activation='relu'))
##    model.add(Dropout(0.2))
#
#    model.add(Dense(1, activation="sigmoid"))
#    optimizer = Adam(lr=0.0003, decay=0.1)
#    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
#
#    return model

def getModel_FCN():
    #Building the model
    model=Sequential()
    
    model.add(Lambda(lambda x: x, input_shape=(75, 75, 3)))
    
    model.add(ZeroPadding2D((1, 1)))
    
    #Conv Layer 1
    model.add(Conv2D(64, kernel_size=(3, 3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Dropout(0.2))

    #Conv Layer 2
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu' ))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))

    #Conv Layer 3
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))

    #Conv Layer 4
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))

    #Flatten the data for upcoming dense layers
    model.add(Flatten())

    #Dense Layers
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    #Dense Layer 2
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    #Sigmoid Layer
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    mypotim=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='binary_crossentropy',
                  optimizer=mypotim,
                  metrics=['accuracy'])
    
    return model

def get_scaled_imgs(df):
    imgs = []
#    R = []
    
    for i, row in df.iterrows():
        #make 75x75 image
        band_1 = np.array(row['band_1']).reshape(75, 75)
        band_2 = np.array(row['band_2']).reshape(75, 75)
        band_3 = band_1 + band_2 # plus since log(x*y) = log(x) + log(y)
        
        # Rescale
        a = (band_1 - band_1.mean()) / (band_1.max() - band_1.min())
        b = (band_2 - band_2.mean()) / (band_2.max() - band_2.min())
        c = (band_3 - band_3.mean()) / (band_3.max() - band_3.min())
 
#        a = stats.zscore(band_1)
#        b = stats.zscore(band_2)
#        c = stats.zscore(band_3)
        
       
#        R0 = get_R(band_1,band_2,band_3)
#        
#        R.append(R0)

        imgs.append(np.dstack((a, b, c)))

    return np.array(imgs)#, R

def get_R(a,b,c):
       
    R_sea = np.mean(a[np.where(a<(a.mean()+a.std()))][0])
    R_a = a.max()-R_sea
    
    R_sea = np.mean(b[np.where(b<(b.mean()+b.std()))][0])
    R_b = b.max()-R_sea
    
    R_sea = np.mean(c[np.where(c<(c.mean()+c.std()))][0])
    R_c = c.max()-R_sea
    
    R = np.hstack((R_a, R_b, R_c))
    
    return R

def get_more_images(imgs):
    
    print(imgs.shape)
    
    more_images = []
    vert_flip_imgs = []
    hori_flip_imgs = []
    
    imgs.shape[0]
    
    for i in range(0,imgs.shape[0]):
        a=imgs[i,:,:,0]
        b=imgs[i,:,:,1]
        c=imgs[i,:,:,2]
        
        av=cv2.flip(a,1)
        ah=cv2.flip(a,0)
        bv=cv2.flip(b,1)
        bh=cv2.flip(b,0)
        cv=cv2.flip(c,1)
        ch=cv2.flip(c,0)
        
        #print(a.shape)
        
        vert_flip_imgs.append(np.dstack((av, bv, cv)))
        hori_flip_imgs.append(np.dstack((ah, bh, ch)))
#        more_images.append(np.dstack((av, bv, cv)))
#        more_images.append(np.dstack((ah, bh, ch)))      
      
    v = np.array(vert_flip_imgs)
    h = np.array(hori_flip_imgs)
    
    print('imgs: ',imgs.shape)
    print('vert: ',v.shape)
    print('hori: ',h.shape)
    
    more_images = np.concatenate((imgs,v,h))
    
    print('more: ',more_images.shape)
    
    return more_images

#def get_filtered_imgs(imgs):
#    
#    fimgs = []
#    
#    for i in range(0,imgs.shape[0]):
#        a=imgs[i,:,:,0]
#        b=imgs[i,:,:,1]
#        c=imgs[i,:,:,2]
#        
#        af = cv2.bilateralFilter(a,5,15,15)
#        bf = cv2.bilateralFilter(b,5,15,15)
#        cf = cv2.bilateralFilter(c,5,15,15)
#        
#        fimgs.append(np.dstack((af,bf,cf)))
#        
#    return np.array(fimgs)

#def random_forest_classifier(features, target):
#    """
#    To train the random forest classifier with features and target data
#    :param features:
#    :param target:
#    :return: trained random forest classifier
#    """
#    clf = RandomForestClassifier()
#    clf.fit(features, target)
#    return clf

# Read the json files into a pandas dataframe
print('Fetching Training Data...')
#with open(INPUT_PATH + 'train.json') as datafile:
#    data = json.load(datafile)
    
df_train = pd.read_json(INPUT_PATH + 'train.json')    
    
#df_train = pd.DataFrame(data)

df_train.inc_angle = df_train.inc_angle.replace('na',0)
idx_tr = np.where(df_train.inc_angle>0)
Ytrain = np.array(df_train['is_iceberg'])
Xtrain = get_scaled_imgs(df_train)

# should we implement a denoising AE?

#train_t = get_scaled_imgs(df_train)
#Xtrain = train_t[0]
#R_train = train_t[1]

Ytrain = Ytrain[idx_tr[0]]
Xtrain = Xtrain[idx_tr[0],...]

# Bilateral filter the images
#Xtrain = get_filtered_imgs(Xtrain)

#inc_angle_train = np.array(df_train['inc_angle'])

#add more images to train on
Xtr_more = get_more_images(Xtrain) 
Ytr_more = np.concatenate((Ytrain,Ytrain,Ytrain))

batch_size = 32
bd=3

#model = get_model_all_CNN(bd)
model = getModel_FCN()
model.summary()

earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
mcp_save = ModelCheckpoint('.mdl_wtsFCN.hdf5', save_best_only=True, monitor='val_loss', mode='min')
tensorboard = TensorBoard(log_dir='../logs', histogram_freq=0, write_graph=True, write_images=False)
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')
model.fit(Xtr_more[:,:,:,0:], Ytr_more, batch_size=batch_size, epochs=50, verbose=1, callbacks=[earlyStopping, mcp_save,tensorboard,reduce_lr_loss], validation_split=0.25)

model.load_weights(filepath = '.mdl_wtsFCN.hdf5')

score = model.evaluate(Xtrain[:,:,:,0:], Ytrain, verbose=2)
print('Train score:', score[0])
print('Train accuracy:', score[1])

print('Fetching Test Data...')  
df_test = pd.read_json(INPUT_PATH + 'test.json')
    

df_test.inc_angle = df_test.inc_angle.replace('na',0)
Xtest = (get_scaled_imgs(df_test))

predA_train = model.predict(Xtrain[:,:,:,0:])
print(predA_train.reshape(predA_train.shape[0]))

predA_test = model.predict(Xtest[:,:,:,0:])
print(predA_test.reshape(predA_test.shape[0]))

submission = pd.DataFrame({'id': df_test["id"], 'is_iceberg': predA_test.reshape((predA_test.shape[0]))})
print(submission.head(10))

submission.to_csv(INPUT_PATH + '20171124_submission_2.csv', index=False)
### ----

### ---
# Okay, we have some predictions, they aren't bad. Can we do better by using some pseudo-labelling?

idx_pred_1 = (np.where(predA_test[:,0]>0.95))
idx_pred_0 = (np.where(predA_test[:,0]<0.05))

Xtrain_pl = np.concatenate((Xtrain,Xtest[idx_pred_1[0],...],Xtest[idx_pred_0[0],...]))
Ytrain_pl = np.concatenate((Ytrain,np.ones(idx_pred_1[0].shape[0]),np.zeros(idx_pred_0[0].shape[0])))

model = getModel_FCN()
model.fit(Xtrain_pl, Ytrain_pl, batch_size=batch_size, epochs=50, verbose=1, callbacks=[earlyStopping, mcp_save,tensorboard,reduce_lr_loss], validation_split=0.25)

model.load_weights(filepath = '.mdl_wtsFCN.hdf5')

score = model.evaluate(Xtrain_pl, Ytrain_pl, verbose=2)
print('Train score:', score[0])
print('Train accuracy:', score[1])

predA_test = model.predict(Xtest)
print(predA_test.reshape(predA_test.shape[0]))

submission = pd.DataFrame({'id': df_test["id"], 'is_iceberg': predA_test.reshape((predA_test.shape[0]))})
print(submission.head(10))

submission.to_csv(INPUT_PATH + '20171124_submission_2_pl.csv', index=False)



