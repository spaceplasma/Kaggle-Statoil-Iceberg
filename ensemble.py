# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 11:35:44 2017

@author: bryacha
"""

import pandas as pd 
import numpy as np 
from scipy.stats.mstats import gmean

print('Fetching Training Data...')
INPUT_PATH='C:\\Users\\bryacha\\projects\\Kaggle\\Iceberg\\data\\processed\\' 
df_train = pd.read_json(INPUT_PATH + 'train.json')    

df_1 = pd.read_csv(INPUT_PATH+'20171211_submission_0.438274669533.csv') #1
df_2 = pd.read_csv(INPUT_PATH+'20171211_submission_0.453489136938.csv') 
df_3 = pd.read_csv(INPUT_PATH+'20171211_submission_0.45163489965.csv') #1
df_4 = pd.read_csv(INPUT_PATH+'20171211_submission_0.430202451092.csv') 
df_5 = pd.read_csv(INPUT_PATH+'20171211_submission_0.433635282857.csv') #1
df_6 = pd.read_csv(INPUT_PATH+'20171123_submission_1_1995.csv') #1
df_7 = pd.read_csv(INPUT_PATH+'20171127_submission_2_pl_2077.csv') #1
df_8 = pd.read_csv(INPUT_PATH+'20171207_submission_0.45214796698.csv') #maybe
df_9 = pd.read_csv(INPUT_PATH+'20171207_submission_0.469067448954.csv') #meh
df_10 = pd.read_csv(INPUT_PATH+'20171211_submission_0.453621749148.csv')
df_11 = pd.read_csv(INPUT_PATH+'20171211_submission_0.456525567966.csv')
df_12 = pd.read_csv(INPUT_PATH+'20171211_submission_0.462092368267.csv')
df_13 = pd.read_csv(INPUT_PATH+'20171211_submission_0.456864461949.csv')
df_14 = pd.read_csv(INPUT_PATH+'20171211_submission_0.443601988711.csv')
df_15 = pd.read_csv(INPUT_PATH+'20171127_submission_3_pl.csv')
df_16 = pd.read_csv(INPUT_PATH+'20171207_submission_ens2.csv') #meh
df_17 = pd.read_csv(INPUT_PATH+'20171126_submission_1_pl_990_968.csv') #1


q = np.array(df_1['is_iceberg'])

#is_iceberg_arr = np.concatenate((np.array(df_1['is_iceberg']), np.array(df_2['is_iceberg']), np.array(df_3['is_iceberg']), np.array(df_4['is_iceberg']), np.array(df_5['is_iceberg']), np.array(df_6['is_iceberg']), np.array(df_7['is_iceberg']), np.array(df_8['is_iceberg']), np.array(df_9['is_iceberg']), np.array(df_10['is_iceberg']), np.array(df_11['is_iceberg']), np.array(df_12['is_iceberg']), np.array(df_13['is_iceberg']), np.array(df_14['is_iceberg']), np.array(df_15['is_iceberg']), np.array(df_16['is_iceberg']), np.array(df_17['is_iceberg'])))
#is_iceberg_arr = np.concatenate((np.array(df_2['is_iceberg']), np.array(df_3['is_iceberg']), np.array(df_4['is_iceberg'])))
is_iceberg_arr = np.concatenate((np.array(df_1['is_iceberg']), np.array(df_3['is_iceberg']), np.array(df_5['is_iceberg']), np.array(df_6['is_iceberg']), np.array(df_7['is_iceberg']), np.array(df_17['is_iceberg'])))

is_iceberg_arr = is_iceberg_arr.reshape(-1,8424)

print(np.corrcoef(is_iceberg_arr))

is_iceberg_arr = np.transpose(is_iceberg_arr)

# Straight mean
ii_arr = np.mean(is_iceberg_arr,axis = 1)

print(np.corrcoef(ii_arr,q))

# geometric Mean
ii_arr = gmean(is_iceberg_arr,axis = 1)

print(np.corrcoef(ii_arr,q))

submission = pd.DataFrame({'id': df_1["id"], 'is_iceberg': ii_arr})
print(submission.head(20))

#submission.to_csv(INPUT_PATH + '20171212_submission_ens3.csv', index=False)


