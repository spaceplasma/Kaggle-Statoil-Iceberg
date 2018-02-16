# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 12:21:13 2018

@author: bryacha
"""
import numpy as np
import pandas as pd
import cv2

from sklearn import svm, metrics
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance

INPUT_PATH='Data\\' 

def lee_filter(img, size):
    
    img_mean = uniform_filter(img, (size, size))
    img_sqr_mean = uniform_filter(img**2, (size, size))
    img_variance = img_sqr_mean - img_mean**2

    overall_variance = variance(img)

    img_weights = img_variance**2 / (img_variance**2 + overall_variance**2)
    img_output = img_mean + img_weights * (img - img_mean)

    return img_output

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
        
    imgs = np.array(imgs)
    
    return imgs

def get_augment(imgs):
      
    more_images = []
#    vert_flip_imgs = []
    hori_flip_imgs = []
    
    imgs.shape[0]
    
    for i in range(0,imgs.shape[0]):
        a = imgs[i,:,:]
        a=imgs[i,:,:,0]
        b=imgs[i,:,:,1]
        c=imgs[i,:,:,2]
        
#        av=cv2.flip(a,1)
        ah=cv2.flip(a,0)
#        bv=cv2.flip(b,1)
        bh=cv2.flip(b,0)
#        cv=cv2.flip(c,1)
        ch=cv2.flip(c,0)
       
#        vert_flip_imgs.append(np.dstack((av, bv, cv)))
        hori_flip_imgs.append(np.dstack((ah, bh, ch)))
      
#    v = np.array(vert_flip_imgs)
    h = np.array(hori_flip_imgs)
    
#    more_images = np.concatenate((imgs,v,h))
    more_images = np.concatenate((imgs,h))
    
    return more_images

print('Fetching Training Data...')
df_train = pd.read_json(INPUT_PATH + 'train.json')    

df_train.inc_angle = df_train.inc_angle.replace('na',0)
idx_tr = np.where(df_train.inc_angle>0)
Ytrain = np.array(df_train['is_iceberg'])
Xtrain = get_scaled_imgs(df_train)
Ytrain = Ytrain[idx_tr[0]]
Xtrain = Xtrain[idx_tr[0],...]

IAtrain = (np.array(df_train['inc_angle']))[idx_tr[0]]

#print('Fetching Test Data...')  
#df_test = pd.read_json(INPUT_PATH + 'test.json', precise_float=True)
#Xtest_all = get_scaled_imgs(df_test)
#IAtest_all = np.array(df_test.inc_angle)

#tr_samples = len(Ytrain)
#Xtrain = Xtrain.reshape(tr_samples,-1)
#
#print('Grid Searching for best hyperparameters')
#parameters = {'C':[0.8, 1, 2, 5],
#              'gamma': [0.008, 0.01, 0.012]}
#
#svc = svm.SVC()
#clf = GridSearchCV(svc, parameters)
#clf.fit(Xtrain,Ytrain)
#
#print(sorted(clf.cv_results_.keys()))

print('Fetching Test Data...')  
df_test = pd.read_json(INPUT_PATH + 'test.json', precise_float=True)
Xtest = get_scaled_imgs(df_test)
Xtest = Xtest.reshape((Xtest.shape)[0],-1)

print('Training and Predicting')

ens = pd.DataFrame({'id': df_test["id"]})

sss = StratifiedShuffleSplit(n_splits=11)
i=0

for train_index, cv_index in sss.split(Xtrain, Ytrain):

    i += 1

    X_train, X_cv = Xtrain[train_index], Xtrain[cv_index]
    y_train, y_cv = Ytrain[train_index], Ytrain[cv_index]
#    ia_train, ia_cv = IAtrain[train_index], IAtrain[cv_index]
    
    #add more images to train on
    Xtr_more = get_augment(X_train) 
    Ytr_more = np.concatenate((y_train,y_train))
#    IAtr_more = np.concatenate((ia_train,ia_train))
#    Ytr_more = np.concatenate((y_train,y_train,y_train))
#    IAtr_more = np.concatenate((ia_train,ia_train,ia_train))

    #add more images to train on
    Xcv_more = get_augment(X_cv) 
    Ycv_more = np.concatenate((y_cv,y_cv))
#    IAcv_more = np.concatenate((ia_cv,ia_cv))

    #SVM
    tr_samples = len(Ytr_more)
    Xtr_more = Xtr_more.reshape(tr_samples,-1)
    cv_samples = len(Ycv_more)
    Xcv_more = Xcv_more.reshape(cv_samples,-1)
   
    #create a basic classifier
    classifier = svm.SVC(gamma=0.012,C=2,kernel='rbf',probability=True)
    
    classifier.fit(Xtr_more,Ytr_more)
    
    pred = classifier.predict(Xcv_more)
    print("Classification report for classifier %s:\n%s\n"
          % (classifier, metrics.classification_report(Ycv_more, pred)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(Ycv_more, pred))
    
    pred_tr = classifier.predict(Xtrain.reshape(len(Ytrain),-1))
#    print("Classification report for classifier %s:\n%s\n"
#          % (classifier, metrics.classification_report(Ytrain, pred_tr)))
#    print("Confusion matrix:\n%s" % metrics.confusion_matrix(Ytrain, pred_tr))
    #print(classifier.score(Xtrain.reshape(len(Ytrain),-1),Ytrain))
    
    tr_acc = metrics.accuracy_score(Ytrain, pred_tr)
    f1_sc = metrics.f1_score(Ytrain, pred_tr)
    
    print('F1-score: ',f1_sc)
    print('Accuracy score: ',tr_acc)
    
    predA_test = classifier.predict_proba(Xtest)
    
    ens['is_iceberg_'+str(i)] = predA_test[:,1]

print(ens.head(25))

# Make an ensemble of all the RF results

fields = ens.iloc[:,1:]

ens['is_iceberg_max'] = fields.max(axis=1)
ens['is_iceberg_min'] = fields.min(axis=1)
ens['is_iceberg_mean'] = fields.mean(axis=1)
ens['is_iceberg_median'] = fields.median(axis=1)

#MinMax + Median Stacking
ens['is_iceberg'] = np.where(np.all(ens.iloc[:,1:-4] > 0.7, axis=1), 
                                    ens['is_iceberg_max'], 
                                    np.where(np.all(ens.iloc[:,1:-4] < 0.3, axis=1),
                                             ens['is_iceberg_min'], 
                                             ens['is_iceberg_median']))
ens[['id', 'is_iceberg']].to_csv(INPUT_PATH + '20180120_ens_svm.csv', 
                                        index=False, float_format='%.6f')

print(ens.head(25))





