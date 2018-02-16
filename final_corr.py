# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 14:19:21 2018

@author: bryacha
"""

import pandas as pd 
import numpy as np 
#from scipy.stats.mstats import gmean
#import matplotlib.pyplot as plt
import os

#def majority_vote(df):
#    
#    df = ((df.round(0)).mean(axis=1)).round(0)
#    
#    return df


INPUT_PATH='Data\\' 
#df_test = pd.read_json(INPUT_PATH + 'test.json', precise_float=True)
##df_ia = df_test['inc_angle']
#
#df_test[['id', 'inc_angle']].to_csv(INPUT_PATH + 'inc_angle.csv')
      
df_ia = pd.read_csv(INPUT_PATH + 'inc_angle.csv', float_precision = 'round_trip')    

idx = (np.where(df_ia['inc_angle'] == round(df_ia['inc_angle'],4)))[0]

sub_path = INPUT_PATH+"\\corr\\"
#sub_path = INPUT_PATH+"\\svm_ens\\"

all_files = os.listdir(sub_path)

# Read and concatenate submissions
outs = [pd.read_csv(os.path.join(sub_path, f), index_col=0) for f in all_files]
print(all_files)

#fix_id = (outs[1])['id']
#fix_prob = ((outs[0])['is_iceberg'])
#fixed = pd.concat([fix_id,fix_prob], ignore_index=True,axis=1)
#fixed.columns = ['id','is_iceberg']
#fixed[['id', 'is_iceberg']].to_csv(sub_path + 'QS_submission.csv',index=False)

ships = []
icebergs = []
for i in range(len(outs)-1):
    if i == 0:
        reals = (np.array(outs[i]))[idx]
#        ships.append(len(((np.array(outs[i]))[idx])[np.where((np.array(outs[i]))[idx]<0.05)]))
#        icebergs.append(len(((np.array(outs[i]))[idx])[np.where((np.array(outs[i]))[idx]>0.95)]))
    else:
        reals = np.hstack((reals,(np.array(outs[i]))[idx]))
#        ships.append(len(((np.array(outs[i]))[idx])[np.where((np.array(outs[i]))[idx]<0.05)]))
#        icebergs.append(len(((np.array(outs[i]))[idx])[np.where((np.array(outs[i]))[idx]>0.95)]))


print(np.corrcoef(reals.T))

ice0 = (np.where(reals[:,0]>0.65))[0]
ice1 = (np.where(reals[:,1]>0.65))[0]
ice2 = (np.where(reals[:,2]>0.65))[0]
ice3 = (np.where(reals[:,3]>0.65))[0]
ship0 = (np.where(reals[:,0]<=0.35))[0]
ship1 = (np.where(reals[:,1]<=0.35))[0]
ship2 = (np.where(reals[:,2]<=0.35))[0]
ship3 = (np.where(reals[:,3]<=0.35))[0]
ice_a = (np.where((np.array(outs[0]))>0.65))[0]
shp_a = (np.where((np.array(outs[0]))<=0.35))[0]
ice_b = (np.where((np.array(outs[1]))>0.65))[0]
shp_b = (np.where((np.array(outs[1]))<=0.35))[0]
ice_c = (np.where((np.array(outs[2]))>0.65))[0]
shp_c = (np.where((np.array(outs[2]))<=0.35))[0]
ice_d = (np.where((np.array(outs[3]))>0.65))[0]
shp_d = (np.where((np.array(outs[3]))<=0.35))[0]
#
print('Intersections',all_files[:-1])
print('All Icebergs: ',len(set(ice0) & set(ice1) & set(ice2) & set(ice3)))
print('All Ships: ',len(set(ship0) & set(ship1) & set(ship2) & set(ship3)))
print('Total set: ',len(set(ice0) & set(ice1) & set(ice2) & set(ice3))+len(set(ship0) & set(ship1) & set(ship2) & set(ship3)),len(reals[:,1]))
print('Icebergs 1: ',len(set(ice0) & set(ship1) & set(ship2) & set(ship3)))
#print(reals[list(set(ice0) & set(ship1) & set(ship2) & set(ship3)),:])
print('Icebergs 2: ',len(set(ice1) & set(ship0) & set(ship2) & set(ship3)))
#print(reals[list(set(ice1) & set(ship0) & set(ship2) & set(ship3)),:])
print('Icebergs 3: ',len(set(ice2) & set(ship0) & set(ship1) & set(ship3)))
#print(reals[list(set(ice2) & set(ship0) & set(ship1) & set(ship3)),:])
print('Icebergs 4: ',len(set(ice3) & set(ship0) & set(ship1) & set(ship2)))
#print(reals[list(set(ice3) & set(ship0) & set(ship1) & set(ship2)),:])
print('Icebergs 1 & 2: ',len(set(ice0) & set(ice1) & set(ship2) & set(ship3)))
#print(reals[list(set(ice0) & set(ice1) & set(ship2) & set(ship3)),:])
print('Icebergs 1 & 3: ',len(set(ice0) & set(ship1) & set(ice2) & set(ship3)))
#print(reals[list(set(ice0) & set(ship1) & set(ice2) & set(ship3)),:])
print('Icebergs 1 & 4: ',len(set(ice0) & set(ship1) & set(ship2) & set(ice3)))
#print(reals[list(set(ice0) & set(ship1) & set(ship2) & set(ice3)),:])
print('Icebergs 2 & 3: ',len(set(ship0) & set(ice1) & set(ice2) & set(ship3)))
#print(reals[list(set(ship0) & set(ice1) & set(ice2) & set(ship3)),:])
print('Icebergs 2 & 4: ',len(set(ice1) & set(ship0) & set(ship2) & set(ice3)))
#print(reals[list(set(ice1) & set(ship0) & set(ship2) & set(ice3)),:])
print('Icebergs 3 & 4: ',len(set(ship0) & set(ship1) & set(ice2) & set(ice3)))
#print(reals[list(set(ship0) & set(ship1) & set(ice2) & set(ice3)),:])
print('Icebergs 1,2,3: ',len(set(ice0) & set(ice1) & set(ice2) & set(ship3)))
print('Icebergs 1,3,4: ',len(set(ice0) & set(ship1) & set(ice2) & set(ice3)))
print('Icebergs 1,2,4: ',len(set(ice0) & set(ice1) & set(ship2) & set(ice3)))
print('Icebergs 2,3,4: ',len(set(ship0) & set(ice1) & set(ice2) & set(ice3)))
#print(reals[list(set(ice0) & set(ice1) & set(ice2) & set(ship3)),:])
#
#



#print(set(ice0) & set(ship1))
#print((set(ice1) & set(ship0)))
#
#print(reals[list(set(ice0) & set(ship1)),0])
#print(reals[list(set(ice0) & set(ship1)),1])
#
#print(reals[list(set(ice1) & set(ship0)),0])
#print(reals[list(set(ice1) & set(ship0)),1])

#int_ice = list(set(ice0) & set(ice1))
#int_shp = list(set(ship0) & set(ship1))
int_all_ice = list(set(ice_a) & set(ice_b) & set(ice_c) & set(ice_d))
int_all_shp = list(set(shp_a) & set(shp_b) & set(shp_d) & set(shp_d))
int_maj_ship_a = list(set(ice_a) & set(shp_b) & set(shp_c) & set(shp_d))
int_maj_ship_b = list(set(ice_b) & set(shp_a) & set(shp_c) & set(shp_d))
int_maj_ship_c = list(set(ice_c) & set(shp_a) & set(shp_b) & set(shp_d))
int_maj_ship_d = list(set(ice_d) & set(shp_a) & set(shp_b) & set(shp_c))
int_maj_ice_a = list(set(shp_a) & set(ice_b) & set(ice_c) & set(ice_d))
int_maj_ice_b = list(set(shp_b) & set(ice_a) & set(ice_c) & set(ice_d))
int_maj_ice_c = list(set(shp_c) & set(ice_a) & set(ice_b) & set(ice_d))
int_maj_ice_d = list(set(shp_d) & set(ice_a) & set(ice_b) & set(ice_c))

concat_sub = pd.concat(outs, axis=1)
cols = list(map(lambda x: "is_iceberg_" + str(x), range(len(concat_sub.columns))))
concat_sub.columns = cols
concat_sub.reset_index(inplace=True)

fields = concat_sub.iloc[:, 1:]
sm_fields = concat_sub.iloc[:, 1:-1]
concat_sub['is_iceberg_max'] = fields.max(axis=1)
idx_max = (np.where(concat_sub['is_iceberg_max'] == 1))[0]


concat_sub['is_iceberg_min'] = fields.min(axis=1)
idx_min = (np.where(concat_sub['is_iceberg_min'] == 0))[0]

concat_sub['is_iceberg_sm_mean'] = sm_fields.mean(axis=1)
concat_sub['is_iceberg_mean'] = fields.mean(axis=1)
concat_sub['is_iceberg_median'] = fields.median(axis=1)

#print(concat_sub['is_iceberg_mean'].head(10))
#print(concat_sub['is_iceberg_sm_mean'].head(10))

#sub_file = np.zeros(len(outs[i])) + (concat_sub['is_iceberg_mean']*0.5+concat_sub['is_iceberg_sm_mean']*0.5)
sub_file = np.zeros(len(outs[i])) + (concat_sub['is_iceberg_4'])

sub_file[int_all_ice] = (concat_sub['is_iceberg_max'])[int_all_ice]
sub_file[int_all_shp] = (concat_sub['is_iceberg_min'])[int_all_shp]
sub_file[int_maj_ship_a] = (concat_sub['is_iceberg_min'])[int_maj_ship_a]
sub_file[int_maj_ship_b] = (concat_sub['is_iceberg_min'])[int_maj_ship_b]
sub_file[int_maj_ship_c] = (concat_sub['is_iceberg_min'])[int_maj_ship_c]
sub_file[int_maj_ship_d] = (concat_sub['is_iceberg_min'])[int_maj_ship_d]
sub_file[int_maj_ice_a] = (concat_sub['is_iceberg_max'])[int_maj_ice_a]
sub_file[int_maj_ice_b] = (concat_sub['is_iceberg_max'])[int_maj_ice_b]
sub_file[int_maj_ice_c] = (concat_sub['is_iceberg_max'])[int_maj_ice_c]
sub_file[int_maj_ice_d] = (concat_sub['is_iceberg_max'])[int_maj_ice_d]

# Fix overflow
idx_max = (np.where(sub_file >= 1))[0]
idx_min = (np.where(sub_file <= 0))[0]

sub_file[idx_min] = 0.000001
sub_file[idx_max] = 0.999999

print(sub_file.head(10))
#
concat_sub['is_iceberg'] = sub_file
concat_sub[['id', 'is_iceberg']].to_csv(sub_path + '20180122_tweak.csv', 
                                        index=False, float_format='%.6f')

from sklearn.metrics import log_loss

rnd_good = np.array(np.around(concat_sub['is_iceberg_4']))
ll_1656 = log_loss(rnd_good[idx],(np.array(concat_sub['is_iceberg_3']))[idx])
ll_sub = log_loss(rnd_good[idx],sub_file[idx])
ll_1401 = log_loss(rnd_good[idx],(np.array(concat_sub['is_iceberg_2']))[idx])
ll_svm = log_loss(rnd_good[idx],(np.array(concat_sub['is_iceberg_1']))[idx])
#ll_1327 = log_loss(rnd_good[idx],(np.array(concat_sub['is_iceberg_2']))[idx])
ll_RF = log_loss(rnd_good[idx],(np.array(concat_sub['is_iceberg_0']))[idx])


#print('Log_loss 1327: ',ll_1327)
print('Log_loss - Sub file: ',ll_sub)
print('Log_loss - 1401: ',ll_1401)
print('Log_loss - svm: ',ll_svm)
print('Log_loss - 1656: ',ll_1656)
print('Log_loss - 1324: ',ll_RF)

