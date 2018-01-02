# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 11:35:44 2017

@author: bryacha
"""

import pandas as pd 
import numpy as np 
from scipy.stats.mstats import gmean
import os

def majority_vote(df):
    
    df = ((df.round(0)).mean(axis=1)).round(0)
    
    return df

print('Fetching Training Data...')
INPUT_PATH='C:\\Users\\bryacha\\projects\\Kaggle\\Iceberg\\data\\processed\\' 
df_train = pd.read_json(INPUT_PATH + 'train.json')    

sub_path = INPUT_PATH+"\\ens2\\"
#sub_path = INPUT_PATH+"\\RF_ens\\"
all_files = os.listdir(sub_path)

# Read and concatenate submissions
outs = [pd.read_csv(os.path.join(sub_path, f), index_col=0) for f in all_files]
print(all_files)
concat_sub = pd.concat(outs, axis=1)
cols = list(map(lambda x: "is_iceberg_" + str(x), range(len(concat_sub.columns))))
concat_sub.columns = cols
concat_sub.reset_index(inplace=True)
#print(concat_sub.head())

# check correlation
print(concat_sub.corr())

fields = concat_sub.iloc[:, 1:]

# get the data fields ready for stacking
concat_sub['is_iceberg_max'] = fields.max(axis=1)
concat_sub['is_iceberg_min'] = fields.min(axis=1)
concat_sub['is_iceberg_mean'] = fields.mean(axis=1)
concat_sub['is_iceberg_median'] = fields.median(axis=1)
concat_sub['is_iceberg_gmean'] = gmean(fields,axis=1)


# set up cutoff threshold for lower and upper bounds, easy to twist 
cutoff_lo = 0.9
cutoff_hi = 0.1

#Mean
concat_sub['is_iceberg'] = concat_sub['is_iceberg_mean']
concat_sub[['id', 'is_iceberg']].to_csv(INPUT_PATH + 'stack_mean.csv', 
                                        index=False, float_format='%.6f')

#Median
concat_sub['is_iceberg'] = concat_sub['is_iceberg_median']
concat_sub[['id', 'is_iceberg']].to_csv(INPUT_PATH + 'stack_median.csv', 
                                        index=False, float_format='%.6f')

#GMean
concat_sub['is_iceberg'] = concat_sub['is_iceberg_gmean']
print(concat_sub.head(10))
concat_sub[['id', 'is_iceberg']].to_csv(INPUT_PATH + 'stack_gmean.csv', 
                                        index=False, float_format='%.6f')


#PushOut + Median
concat_sub['is_iceberg'] = np.where(np.all(concat_sub.iloc[:,1:-6] > cutoff_lo, axis=1), 1, 
                                    np.where(np.all(concat_sub.iloc[:,1:] < cutoff_hi, axis=1),
                                             0, concat_sub['is_iceberg_median']))
concat_sub[['id', 'is_iceberg']].to_csv(INPUT_PATH + 'stack_pushout_median.csv', 
                                        index=False, float_format='%.6f')

#MinMax + Mean Stacking
concat_sub['is_iceberg'] = np.where(np.all(concat_sub.iloc[:,1:-6] > cutoff_lo, axis=1), 
                                    concat_sub['is_iceberg_max'], 
                                    np.where(np.all(concat_sub.iloc[:,1:] < cutoff_hi, axis=1),
                                             concat_sub['is_iceberg_min'], 
                                             concat_sub['is_iceberg_mean']))
concat_sub[['id', 'is_iceberg']].to_csv(INPUT_PATH + 'stack_minmax_mean.csv', 
                                        index=False, float_format='%.6f')

#MinMax + Median Stacking
concat_sub['is_iceberg'] = np.where(np.all(concat_sub.iloc[:,1:-6] > cutoff_lo, axis=1), 
                                    concat_sub['is_iceberg_max'], 
                                    np.where(np.all(concat_sub.iloc[:,1:] < cutoff_hi, axis=1),
                                             concat_sub['is_iceberg_min'], 
                                             concat_sub['is_iceberg_median']))
concat_sub[['id', 'is_iceberg']].to_csv(INPUT_PATH + 'stack_minmax_median.csv', 
                                        index=False, float_format='%.6f')

#MinMax + Geometric Mean Stacking
concat_sub['is_iceberg'] = np.where(np.all(concat_sub.iloc[:,1:-6] > cutoff_lo, axis=1), 
                                    concat_sub['is_iceberg_max'], 
                                    np.where(np.all(concat_sub.iloc[:,1:] < cutoff_hi, axis=1),
                                             concat_sub['is_iceberg_min'], 
                                             concat_sub['is_iceberg_gmean']))
concat_sub[['id', 'is_iceberg']].to_csv(INPUT_PATH + 'stack_minmax_gmean.csv', 
                                        index=False, float_format='%.6f')

#Majority Vote
concat_sub['is_iceberg'] = majority_vote(concat_sub.iloc[:,1:-6])
concat_sub[['id', 'is_iceberg']].to_csv(INPUT_PATH + 'stack_majority_vote.csv', 
                                        index=False, float_format='%.6f')


#MinMax + BestBase Stacking
## ??


