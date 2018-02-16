# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 11:35:44 2017

@author: bryacha
"""

import pandas as pd 
import numpy as np 
from scipy.stats.mstats import gmean
import matplotlib.pyplot as plt
import os

def majority_vote(df):
    
    df = ((df.round(0)).mean(axis=1)).round(0)
    
    return df

#print('Fetching Training Data...')
INPUT_PATH='C:\\Users\\bryacha\\projects\\Kaggle\\Iceberg\\data\\processed\\' 
#df_train = pd.read_json(INPUT_PATH + 'train.json')
df_ia = pd.read_csv(INPUT_PATH + 'inc_angle.csv', float_precision = 'round_trip')    
#print(df_ia.head())

idx = (np.where(df_ia['inc_angle'] == round(df_ia['inc_angle'],4)))[0]
#print(len(idx))

#sub_path = INPUT_PATH+"\\test\\"
#sub_path = INPUT_PATH+"\\ens2\\"
sub_path = INPUT_PATH+"\\ens4\\"
#sub_path = INPUT_PATH+"\\RF_ens\\"
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
for i in range(len(outs)):
    if i == 0:
        reals = (np.array(outs[i]))[idx]
        ships.append(len(((np.array(outs[i]))[idx])[np.where((np.array(outs[i]))[idx]<0.05)]))
        icebergs.append(len(((np.array(outs[i]))[idx])[np.where((np.array(outs[i]))[idx]>0.95)]))
        ice0 = (np.where((np.array(outs[i]))[idx]>0.9))[0]
        ship0 = (np.where((np.array(outs[i]))[idx]<0.1))[0]
        ice_a = (np.where((np.array(outs[i]))>0.9))[0]
        shp_a = (np.where((np.array(outs[i]))<0.1))[0]
    else:
        reals = np.hstack((reals,(np.array(outs[i]))[idx]))
        ships.append(len(((np.array(outs[i]))[idx])[np.where((np.array(outs[i]))[idx]<0.05)]))
        icebergs.append(len(((np.array(outs[i]))[idx])[np.where((np.array(outs[i]))[idx]>0.95)]))
        ice1 = (np.where((np.array(outs[i]))[idx]>0.9))[0]
        ship1 = (np.where((np.array(outs[i]))[idx]<0.1))[0]
        ice_b = (np.where((np.array(outs[i]))>0.9))[0]
        shp_b = (np.where((np.array(outs[i]))<0.1))[0]

#plt.figure(1)
#plt.plot(reals[ice0,0],'k*')
#plt.plot(reals[ice1,2],'r*')
#plt.show
#
#plt.figure(2)
#plt.plot(reals[ship0,0],'k*')
#plt.plot(reals[ship1,2],'r*')
#plt.show
# 
#plt.figure(3)
#plt.plot(reals[:,0],'k.')
##plt.plot(reals[:,1],'bo')
#plt.plot(reals[:,1],'r.')
##plt.plot((reals[:,0]+reals[:,2])/2,'g*')
#plt.ylim(0.05,0.95)
#plt.show
        

#print(ice0[0][0])
#print(ice1[0][0])
        
ice0 = (np.where(reals[:,0]>0.8))[0]
ice1 = (np.where(reals[:,1]>0.8))[0]
ice2 = (np.where(reals[:,3]>0.8))[0]
ice_a = (np.where((np.array(outs[0]))>0.8))[0]
ice_b = (np.where((np.array(outs[1]))>0.8))[0]
ice_c = (np.where((np.array(outs[3]))>0.8))[0]

ship0 = (np.where(reals[:,0]<0.2))[0]
ship1 = (np.where(reals[:,1]<0.2))[0]
ship2 = (np.where(reals[:,3]<0.2))[0]
shp_a = (np.where((np.array(outs[0]))<0.2))[0]
shp_b = (np.where((np.array(outs[1]))<0.2))[0]
shp_c = (np.where((np.array(outs[3]))<0.2))[0]

print('Intersections',all_files[0],all_files[1],all_files[3])
print(len(set(ice0) & set(ice1) & set(ice2)))
print(len(set(ship0) & set(ship1) & set(ship2)))
#print(len(set(ice0) & set(ship1)))
#print(len(set(ice1) & set(ship0)))
#print(set(ice0) & set(ship1))
#print((set(ice1) & set(ship0)))
#
#print(reals[list(set(ice0) & set(ship1)),0])
#print(reals[list(set(ice0) & set(ship1)),1])

int_ice = list(set(ice0) & set(ice1) & set(ice2))
int_shp = list(set(ship0) & set(ship1) & set(ship2))
int_all_ice = list(set(ice_a) & set(ice_b) & set(ice_c))
int_all_shp = list(set(shp_a) & set(shp_b) & set(shp_c))

#print('Maybe iceberg: ',len(np.where(reals[:,0]>0.5)[0])-len(ice0),len(np.where(reals[:,1]>0.5)[0])-len(ice1))
#print('Maybe ships: ',len(np.where(reals[:,0]<0.5)[0])-len(ship0),len(np.where(reals[:,1]<0.5)[0])-len(ship1))
#
#print('What are these? ',len(np.where(reals[:,0]>0.33)[0])-len(np.where(reals[:,0]>0.66)[0]),len(np.where(reals[:,1]>0.33)[0])-len(np.where(reals[:,1]>0.66)[0]))
#
#print(len(set(np.where(reals[:,0]>0.75)[0]) & set(np.where(reals[:,1]>0.75)[0])),len(np.where(reals[:,0]>0.75)[0]),len(np.where(reals[:,1]>0.75)[0]))
#print(len(set(np.where(reals[:,0]<0.25)[0]) & set(np.where(reals[:,1]<0.25)[0])),len(np.where(reals[:,0]<0.25)[0]),len(np.where(reals[:,1]<0.25)[0]))
#print(len(set(np.where(reals[:,0]>0.75)[0]) & set(np.where(reals[:,1]<0.25)[0])),len(np.where(reals[:,0]>0.75)[0]),len(np.where(reals[:,1]<0.25)[0]))
#print(len(set(np.where(reals[:,0]<0.25)[0]) & set(np.where(reals[:,1]>0.75)[0])),len(np.where(reals[:,0]<0.25)[0]),len(np.where(reals[:,1]>0.75)[0]))

#print("Icebergs / Ships: ",icebergs, ships)

print(np.corrcoef(reals.T))

concat_sub = pd.concat(outs, axis=1)

cols = list(map(lambda x: "is_iceberg_" + str(x), range(len(concat_sub.columns))))
concat_sub.columns = cols
concat_sub.reset_index(inplace=True)
#print(concat_sub.head())

# check correlation
#print(concat_sub.corr())

fields = concat_sub.iloc[:, 1:]

# get the data fields ready for stacking
concat_sub['is_iceberg_max'] = fields.max(axis=1)
concat_sub['is_iceberg_min'] = fields.min(axis=1)
concat_sub['is_iceberg_mean'] = fields.mean(axis=1)
concat_sub['is_iceberg_median'] = fields.median(axis=1)
concat_sub['is_iceberg_gmean'] = gmean(fields,axis=1)
concat_sub['is_iceberg_wgt_med'] = fields.median(axis=1)*0.8+fields['is_iceberg_3']*0.2

sub_file = np.zeros(len(outs[i])) + concat_sub['is_iceberg_wgt_med']
sub_file[int_all_ice] = (concat_sub['is_iceberg_max'])[int_all_ice]
sub_file[int_all_shp] = (concat_sub['is_iceberg_min'])[int_all_shp]

print(sub_file.head(10))

#sub_file = np.zeros(len(outs[i])) + concat_sub['is_iceberg_mean']
#sub_file[int_all_ice] = (concat_sub['is_iceberg_max'])[int_all_ice]
#sub_file[int_all_shp] = (concat_sub['is_iceberg_min'])[int_all_shp]
#
#print(sub_file.head(10))


# set up cutoff threshold for lower and upper bounds, easy to twist 
#cutoff_lo = 0.9
#cutoff_hi = 0.1

#Test
#concat_sub['is_iceberg'] = sub_file
#concat_sub[['id', 'is_iceberg']].to_csv(sub_path + 'test.csv', 
#                                        index=False, float_format='%.6f')


#Mean
#concat_sub['is_iceberg'] = concat_sub['is_iceberg_mean']
#concat_sub[['id', 'is_iceberg']].to_csv(sub_path + 'stack_mean.csv', 
#                                        index=False, float_format='%.6f')
#
##Median
#concat_sub['is_iceberg'] = concat_sub['is_iceberg_median']
#concat_sub[['id', 'is_iceberg']].to_csv(sub_path + 'stack_median.csv', 
#                                        index=False, float_format='%.6f')

#GMean
#concat_sub['is_iceberg'] = concat_sub['is_iceberg_gmean']
#print(concat_sub.head(10))
#concat_sub[['id', 'is_iceberg']].to_csv(sub_path + 'stack_gmean.csv', 
#                                        index=False, float_format='%.6f')


#PushOut + Median
#concat_sub['is_iceberg'] = np.where(np.all(concat_sub.iloc[:,1:-6] > cutoff_lo, axis=1), 1, 
#                                    np.where(np.all(concat_sub.iloc[:,1:] < cutoff_hi, axis=1),
#                                             0, concat_sub['is_iceberg_median']))
#concat_sub[['id', 'is_iceberg']].to_csv(sub_path + 'stack_pushout_median.csv', 
#                                        index=False, float_format='%.6f')

#MinMax + Mean Stacking
#concat_sub['is_iceberg'] = np.where(np.all(concat_sub.iloc[:,1:-6] > cutoff_lo, axis=1), 
#                                    concat_sub['is_iceberg_max'], 
#                                    np.where(np.all(concat_sub.iloc[:,1:] < cutoff_hi, axis=1),
#                                             concat_sub['is_iceberg_min'], 
#                                             concat_sub['is_iceberg_mean']))
#concat_sub[['id', 'is_iceberg']].to_csv(sub_path + 'stack_minmax_mean.csv', 
#                                        index=False, float_format='%.6f')

#MinMax + Median Stacking
#concat_sub['is_iceberg'] = np.where(np.all(concat_sub.iloc[:,1:-6] > cutoff_lo, axis=1), 
#                                    concat_sub['is_iceberg_max'], 
#                                    np.where(np.all(concat_sub.iloc[:,1:-6] < cutoff_hi, axis=1),
#                                             concat_sub['is_iceberg_min'], 
#                                             concat_sub['is_iceberg_median']))
#concat_sub[['id', 'is_iceberg']].to_csv(sub_path + 'stack_minmax_median.csv', 
#                                        index=False, float_format='%.6f')

#MinMax + Geometric Mean Stacking
#concat_sub['is_iceberg'] = np.where(np.all(concat_sub.iloc[:,1:-6] > cutoff_lo, axis=1), 
#                                    concat_sub['is_iceberg_max'], 
#                                    np.where(np.all(concat_sub.iloc[:,1:] < cutoff_hi, axis=1),
#                                             concat_sub['is_iceberg_min'], 
#                                             concat_sub['is_iceberg_gmean']))
#concat_sub[['id', 'is_iceberg']].to_csv(sub_path + 'stack_minmax_gmean.csv', 
#                                        index=False, float_format='%.6f')

#Majority Vote
#concat_sub['is_iceberg'] = majority_vote(concat_sub.iloc[:,1:-6])*concat_sub['is_iceberg_max']
#concat_sub[['id', 'is_iceberg']].to_csv(sub_path + 'stack_majority_vote.csv', 
#                                        index=False, float_format='%.6f')


#MinMax + BestBase Stacking
## ??


