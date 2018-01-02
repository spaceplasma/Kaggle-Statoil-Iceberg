# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 10:56:30 2017

@author: bryacha
"""
import pandas as pd 
import numpy as np 
import cv2

from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance
from matplotlib import pyplot as plt

# Required Python Packages
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit

INPUT_PATH='C:\\Users\\bryacha\\projects\\Kaggle\\Iceberg\\data\\processed\\' 

def get_scaled_imgs(df):
    imgs = []
    
    for i, row in df.iterrows():
        #make 75x75 image
        band_1 = np.array(row['band_1']).reshape(75, 75)
        band_2 = np.array(row['band_2']).reshape(75, 75)
        band_3 = band_1 + band_2 # plus since log(x*y) = log(x) + log(y)
        
        band_1 = lee_filter(band_1,3)
        band_2 = lee_filter(band_2,3)
        band_3 = lee_filter(band_3,3)
        
        imgs.append(np.dstack((band_1, band_2, band_3)))

    return np.array(imgs)

def lee_filter(img, size):
    
    img_mean = uniform_filter(img, (size, size))
    img_sqr_mean = uniform_filter(img**2, (size, size))
    img_variance = img_sqr_mean - img_mean**2

    overall_variance = variance(img)

    img_weights = img_variance**2 / (img_variance**2 + overall_variance**2)
    img_output = img_mean + img_weights * (img - img_mean)

    return img_output

def get_R(imgs):
    
    R = []
    
    for i in range(0,imgs.shape[0]):
        a = imgs[i,:,:,0]
        b = imgs[i,:,:,1]
        c = imgs[i,:,:,2]
           
        R_sea = np.mean(a[np.where(a<(a.mean()+a.std()))][0])
        R.append(a.max()-R_sea)
        
        R_sea = np.mean(b[np.where(b<(b.mean()+b.std()))][0])
        R.append(b.max()-R_sea)
        
        R_sea = np.mean(c[np.where(c<(c.mean()+c.std()))][0])
        R.append(c.max()-R_sea)
        
    return np.array(R).reshape(-1,3)

def random_forest_classifier(features, target):
    """
    To train the random forest classifier with features and target data
    :param features:
    :param target:
    :return: trained random forest classifier
    """
    clf = RandomForestClassifier(n_estimators = 250)
    clf.fit(features, target)
    return clf

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

print('Fetching Training Data...')   
df_train = pd.read_json(INPUT_PATH + 'train.json')    

df_train.inc_angle = df_train.inc_angle.replace('na',0)
idx_tr = np.where(df_train.inc_angle>35)
Ytrain = np.array(df_train['is_iceberg'])
Xtrain = get_scaled_imgs(df_train)
print(Xtrain.shape)
Xtrain = Xtrain[idx_tr[0],...]
Ytrain = Ytrain[idx_tr[0]]

#Xtrain = get_augment(Xtrain)
#Ytrain = np.concatenate((Ytrain,Ytrain,Ytrain))

R = get_R(Xtrain)
inc_angle = np.array(df_train['inc_angle'])
inc_angle = np.array(inc_angle[idx_tr[0]]).reshape(-1,1)
#inc_angle = np.concatenate((inc_angle,inc_angle,inc_angle))

#print(R.shape)
#print(inc_angle.shape)

Xtr = np.hstack((R,inc_angle))

print('Fetching Test Data...')  
df_test = pd.read_json(INPUT_PATH + 'test.json')
df_test.inc_angle = df_test.inc_angle.replace('na',0)
Xtest = (get_scaled_imgs(df_test))
R_test = get_R(Xtest)
inc_angle_test = np.array(df_test['inc_angle'])
inc_angle_test = np.array(inc_angle_test).reshape(-1,1)

Xtest = np.hstack((R_test,inc_angle_test))

sss = StratifiedShuffleSplit(n_splits=10,test_size=0.2)

for train_index, cv_index in sss.split(Xtr, Ytrain):
    X_train, X_cv = Xtr[train_index], Xtr[cv_index]
    Y_train, Y_cv = Ytrain[train_index], Ytrain[cv_index]
    
    # Create random forest classifier instance
    trained_model = random_forest_classifier(X_train, Y_train)
    #print("Trained model :: ", trained_model)
    predictions = trained_model.predict(X_cv)
    pred_prob = trained_model.predict_proba(X_cv)

    print("Train Accuracy :: ", accuracy_score(Y_train, trained_model.predict(X_train)))
    print("CV Accuracy  :: ", accuracy_score(Y_cv, predictions))
    CM = confusion_matrix(Y_cv, predictions)
    TPR = CM[1,1]/np.sum(CM[1,:])
    Prec = CM[1,1]/np.sum(CM[:,1])
    Fscore = 2*(Prec*TPR)/(Prec+TPR)
    print('Recall / TPR : ',TPR)
    print('Precision : ',Prec)
    print('F-score : ',Fscore)
    print("Confusion matrix ", CM)

    predRF_test_prob = trained_model.predict_proba(Xtest)
    predRF_test = trained_model.predict(Xtest)

    submission = pd.DataFrame({'id': df_test["id"], 'is_iceberg': predRF_test})
    #print(submission.head(10))
    break
    
#    submission.to_csv(INPUT_PATH + '\\RF_ens\\20171218_RF_'+str(Fscore)+'.csv', index=False)


#idx_ship = np.where(Ytrain < 1)[0]
#idx_ice = np.where(Ytrain == 1)[0]
#plt.plot(inc_angle[idx_ship],R[idx_ship,0],'bo',inc_angle[idx_ice],R[idx_ice,0],'cs')
#plt.show()
#
#plt.plot(inc_angle[idx_ship],R[idx_ship,1],'go',inc_angle[idx_ice],R[idx_ice,1],'rs')
#plt.show()
#
#plt.plot(inc_angle[idx_ship],R[idx_ship,2],'co',inc_angle[idx_ice],R[idx_ice,2],'rs')
#plt.show()

plt.imshow(Xtrain[4,:,:,0])
plt.show()
print(Xtrain[4,32,:,0])

plt.imshow(Xtrain[4,:,:,1])
plt.show()
print(Xtrain[4,32,:,1])

plt.imshow(Xtrain[4,:,:,2])
plt.show()
print(Xtrain[4,32,:,2])

#Ytrain = Ytrain[idx_tr[0]]
#Xtrain = Xtrain[idx_tr[0],...]


