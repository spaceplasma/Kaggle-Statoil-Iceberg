# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 12:38:04 2018

@author: bryacha
"""

import numpy as np
import pandas as pd
import cv2

from sklearn import metrics
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance
from sklearn.ensemble import RandomForestClassifier

INPUT_PATH='Data\\' 

# Utility function to report best scores
def report(results, n_top=5):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

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
        #band_3 = band_1 / band_2 # plus since log(x*y) = log(x) + log(y)
                        
        band_1 = lee_filter(band_1,4)
        band_2 = lee_filter(band_2,4)
        #band_3 = lee_filter(band_3,4)
               
        # Rescale
        a = (band_1 - band_1.mean()) / (band_1.max() - band_1.min())
        b = (band_2 - band_2.mean()) / (band_2.max() - band_2.min())
        #c = (band_3 - band_3.mean()) / (band_3.max() - band_3.min())
             
#        imgs.append(np.dstack((a, b, c)))
        imgs.append(np.dstack((a, b)))
        
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
#        c=imgs[i,:,:,2]
        
#        av=cv2.flip(a,1)
        ah=cv2.flip(a,0)
#        bv=cv2.flip(b,1)
        bh=cv2.flip(b,0)
#        cv=cv2.flip(c,1)
#        ch=cv2.flip(c,0)
       
#        vert_flip_imgs.append(np.dstack((av, bv, cv)))
#        hori_flip_imgs.append(np.dstack((ah, bh, ch)))
        hori_flip_imgs.append(np.dstack((ah, bh)))
      
#    v = np.array(vert_flip_imgs)
    h = np.array(hori_flip_imgs)
    
#    more_images = np.concatenate((imgs,v,h))
    more_images = np.concatenate((imgs,h))
    
    return more_images

def random_forest_classifier(features, target):
    """
    To train the random forest classifier with features and target data
    :param features:
    :param target:
    :return: trained random forest classifier
    """
    clf = RandomForestClassifier(n_estimators=100, criterion = 'entropy', bootstrap = False, min_samples_leaf = 10, min_samples_split = 10)
    clf.fit(features, target)
    return clf


print('Fetching Training Data...')
df_train = pd.read_json(INPUT_PATH + 'train.json')    

df_train.inc_angle = df_train.inc_angle.replace('na',0)
idx_tr = np.where(df_train.inc_angle>0)
Ytrain = np.array(df_train['is_iceberg'])
Xtrain = get_scaled_imgs(df_train)
Ytrain = Ytrain[idx_tr[0]]
Xtrain = Xtrain[idx_tr[0],...]

IAtrain = (np.array(df_train['inc_angle']))[idx_tr[0]]

print('Fetching Test Data...')  
df_test = pd.read_json(INPUT_PATH + 'test.json', precise_float=True)
Xtest = get_scaled_imgs(df_test)
Xtest = Xtest.reshape((Xtest.shape)[0],-1)

### -----

# GridSearchCV
#print('Grid Searching')
## use a full grid over all parameters
#param_grid = {"max_depth": [None],
#              "min_samples_split": [10, 12, 15],
#              "min_samples_leaf": [7, 10],
#              "bootstrap": [False],
#              "criterion": ["gini", "entropy"]}
#
#clf = RandomForestClassifier(n_estimators=100)
#
## run grid search
#grid_search = GridSearchCV(clf, param_grid=param_grid)
#grid_search.fit(Xtrain.reshape(len(Ytrain),-1), Ytrain)
#report(grid_search.cv_results_)

#
# Best parameters from gridserch:
# "min_samples_split": 10
# "min_samples_leaf": 10
# "max_depth": [None]
# bootstrap": [False]
# "criterion": "entropy"
#

## -----

print('Training and Predicting')

sss = StratifiedShuffleSplit(n_splits=21)

i=0

pred_df = pd.DataFrame({'id': df_test["id"]})

for train_index, cv_index in sss.split(Xtrain, Ytrain):

    i += 1

    X_train, X_cv = Xtrain[train_index], Xtrain[cv_index]
    y_train, y_cv = Ytrain[train_index], Ytrain[cv_index]
    ia_train, ia_cv = IAtrain[train_index], IAtrain[cv_index]
    
    #add more images to train on
    Xtr_more = get_augment(X_train) 
    Ytr_more = np.concatenate((y_train,y_train))
    IAtr_more = np.concatenate((ia_train,ia_train))
#    Ytr_more = np.concatenate((y_train,y_train,y_train))
#    IAtr_more = np.concatenate((ia_train,ia_train,ia_train))

    #add more images to train on
    Xcv_more = get_augment(X_cv) 
    Ycv_more = np.concatenate((y_cv,y_cv))
    IAcv_more = np.concatenate((ia_cv,ia_cv))

    #SVM
    tr_samples = len(Ytr_more)
    Xtr_more = Xtr_more.reshape(tr_samples,-1)
    cv_samples = len(Ycv_more)
    Xcv_more = Xcv_more.reshape(cv_samples,-1)
    
    trained_model = random_forest_classifier(Xtr_more, Ytr_more)
    
#    print("Trained model :: ", trained_model)
    pred_tr = trained_model.predict(Xtrain.reshape(len(Ytrain),-1))
#    print("Classification report for classifier %s:\n%s\n"
#      % (random_forest_classifier, metrics.classification_report(Ytrain, pred_tr)))
#    print("Confusion matrix:\n%s" % metrics.confusion_matrix(Ytrain, pred_tr))

    tr_acc = metrics.accuracy_score(Ytrain, pred_tr)
    f1_sc = metrics.f1_score(Ytrain, pred_tr)
    
    print('F1-score: ',f1_sc)
    print('Accuracy score: ',tr_acc)
    
    predA_test = trained_model.predict_proba(Xtest)
    
    #print(pred_df.head(10))
      
    pred_df['is_iceberg_'+str(i)] = predA_test[:,1]

    #break
    
# Make an ensemble of all the RF results

fields = pred_df.iloc[:,1:]

pred_df['is_iceberg_max'] = fields.max(axis=1)
pred_df['is_iceberg_min'] = fields.min(axis=1)
pred_df['is_iceberg_mean'] = fields.mean(axis=1)
pred_df['is_iceberg_median'] = fields.median(axis=1)

#MinMax + Median Stacking
pred_df['is_iceberg'] = np.where(np.all(pred_df.iloc[:,1:-4] > 0.7, axis=1), 
                                    pred_df['is_iceberg_max'], 
                                    np.where(np.all(pred_df.iloc[:,1:-4] < 0.3, axis=1),
                                             pred_df['is_iceberg_min'], 
                                             pred_df['is_iceberg_median']))
pred_df[['id', 'is_iceberg']].to_csv(INPUT_PATH + '20180120_ens_RF.csv', 
                                        index=False, float_format='%.6f')

print(pred_df.head(25))


 #   submission = pd.DataFrame({'id': df_test["id"], 'is_iceberg': predA_test[:,1]})
    #print(submission.head(10))

    


    #submission.to_csv(INPUT_PATH + '20180118_'+str(tr_acc)+'_'+str(f1_sc)+'_RF.csv', index=False)


#    break