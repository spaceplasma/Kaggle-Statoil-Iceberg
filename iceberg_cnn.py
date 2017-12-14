# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 16:08:32 2017

@author: bryacha
"""

#import json
import pandas as pd 
import numpy as np 
import cv2
np.random.seed(1337)

#import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Lambda, Activation, Input
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, UpSampling2D#, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
#from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
#from keras.utils import np_utils
#from sklearn.ensemble import RandomForestClassifier
#from scipy import stats
#from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance

INPUT_PATH='C:\\Users\\bryacha\\projects\\Kaggle\\Iceberg\\data\\processed\\' 

def lee_filter(img, size):
    
    img_mean = uniform_filter(img, (size, size))
    img_sqr_mean = uniform_filter(img**2, (size, size))
    img_variance = img_sqr_mean - img_mean**2

    overall_variance = variance(img)

    img_weights = img_variance**2 / (img_variance**2 + overall_variance**2)
    img_output = img_mean + img_weights * (img - img_mean)

    return img_output

def getModel(shp):
    #Building the model
    model=Sequential()
    
    model.add(Lambda(lambda x: x, input_shape=(shp)))
    
#    model.add(ZeroPadding2D((1, 1)))
    
    #Conv Layer 1
    model.add(Conv2D(64, kernel_size=(3, 3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    #Conv Layer 2
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    #Conv Layer 3
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    #Conv Layer 4
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))   
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))
    
    #Flatten the data for upcoming dense layers
    model.add(Flatten())

    #Dense Layers
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    #model.add(BatchNormalization())

    #Dense Layer 2
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    #model.add(BatchNormalization())

    #Sigmoid Layer
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    mypotim=Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='binary_crossentropy',
                  optimizer=mypotim,
                  metrics=['accuracy'])
    
    return model

def get_scaled_imgs(df):
    imgs = []
    
    for i, row in df.iterrows():
        band_1 = np.array(row['band_1']).reshape(75, 75)
        band_2 = np.array(row['band_2']).reshape(75, 75)
        band_3 = band_1 / band_2 # plus since log(x*y) = log(x) + log(y)
        
        band_1 = lee_filter(band_1,4)
        band_2 = lee_filter(band_2,4)
        band_3 = lee_filter(band_3,4)
        
        # Rescale
        a = (band_1 - band_1.mean()) / (band_1.max() - band_1.min())
        b = (band_2 - band_2.mean()) / (band_2.max() - band_2.min())
        c = (band_3 - band_3.mean()) / (band_3.max() - band_3.min())
             
        imgs.append(np.dstack((a, b, c)))

    return np.array(imgs)

def get_augment(imgs):
      
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
        
        vert_flip_imgs.append(np.dstack((av, bv, cv)))
        hori_flip_imgs.append(np.dstack((ah, bh, ch)))
      
    v = np.array(vert_flip_imgs)
    h = np.array(hori_flip_imgs)
    
    more_images = np.concatenate((imgs,v,h))
    
    return more_images

# Read the json files into a pandas dataframe
print('Fetching Training Data...')
df_train = pd.read_json(INPUT_PATH + 'train.json')    

print('Fetching Test Data...')  
df_test = pd.read_json(INPUT_PATH + 'test.json')

df_train.inc_angle = df_train.inc_angle.replace('na',0)
idx_tr = np.where(df_train.inc_angle>0)
Ytrain = np.array(df_train['is_iceberg'])
Xtrain = get_scaled_imgs(df_train)
Ytrain = Ytrain[idx_tr[0]]
Xtrain = Xtrain[idx_tr[0],...]

df_test.inc_angle = df_test.inc_angle.replace('na',0)
Xtest = (get_scaled_imgs(df_test))
idx_test = np.where(df_test.inc_angle>0)
Xtest0 = Xtest[idx_test[0],...]

sss = StratifiedShuffleSplit(n_splits=5,test_size=0.2)
i=0

for train_index, cv_index in sss.split(Xtrain, Ytrain):

    X_train, X_cv = Xtrain[train_index], Xtrain[cv_index]
    y_train, y_cv = Ytrain[train_index], Ytrain[cv_index]

    #add more images to train on
    Xtr_more = get_augment(X_train) 
    Ytr_more = np.concatenate((y_train,y_train,y_train))

    #add more images to train on
    Xcv_more = get_augment(X_cv) 
    Ycv_more = np.concatenate((y_cv,y_cv,y_cv))
    
    batch_size = 32
    
    model = getModel(Xtr_more.shape[1:])
    #model.summary()
    
    earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
    mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
    tensorboard = TensorBoard(log_dir='../logs', histogram_freq=0, write_graph=True, write_images=False)
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.05, patience=5, verbose=1, epsilon=1e-4, mode='min')
    model.fit(Xtr_more, Ytr_more, batch_size=batch_size, epochs=30, verbose=0, callbacks=[earlyStopping, mcp_save,tensorboard,reduce_lr_loss], validation_data=(Xcv_more,Ycv_more))
    
    model.load_weights(filepath = '.mdl_wts.hdf5')
    
    score = model.evaluate(Xcv_more, Ycv_more, verbose=2)
    print('Pass:',i+1)
    print('CV loss:', score[0])
    print('CV accuracy:', score[1])

    pt = model.predict(Xcv_more)
    mse = (np.mean((pt-Ycv_more)**2))
    print('CV MSE: ', mse)

    score = model.evaluate(Xtrain, Ytrain, verbose=2)
    print('Train loss:', score[0])
    print('Train accuracy:', score[1])

    pt = model.predict(Xtrain)
    mse = (np.mean((pt-Ytrain)**2))
    print('Train MSE: ', mse)

    
    predA_test = model.predict(Xtest)    

    submission = pd.DataFrame({'id': df_test["id"], 'is_iceberg': predA_test.reshape((predA_test.shape[0]))})
    #print(submission.head(10))
    
    submission.to_csv(INPUT_PATH + '20171214_'+str(score[0])+'_'+str(score[1])+'.csv', index=False)




