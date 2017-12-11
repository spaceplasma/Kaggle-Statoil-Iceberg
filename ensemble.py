# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 11:35:44 2017

@author: bryacha
"""

import pandas as pd 
import numpy as np 

print('Fetching Training Data...')
INPUT_PATH='C:\\Users\\bryacha\\projects\\Kaggle\\Iceberg\\data\\processed\\' 
df_train = pd.read_json(INPUT_PATH + 'train.json')    

df_1 = pd.read_csv(INPUT_PATH+'20171127_submission_3_pl_1656.csv') #75
#df_2 = pd.read_csv(INPUT_PATH+'20171207_submission_0.410927391407.csv') #28
#df_3 = pd.read_csv(INPUT_PATH+'20171207_submission_0.418098104904.csv') #28
#df_4 = pd.read_csv(INPUT_PATH+'20171207_submission_0.414669429683.csv') #28
df_5 = pd.read_csv(INPUT_PATH+'20171207_submission_ens_28.csv') #75
df_6 = pd.read_csv(INPUT_PATH+'20171123_submission_1_1995.csv') #75
df_7 = pd.read_csv(INPUT_PATH+'20171127_submission_2_pl_2077.csv') #75
df_8 = pd.read_csv(INPUT_PATH+'20171207_submission_0.45214796698.csv')
df_9 = pd.read_csv(INPUT_PATH+'20171207_submission_0.469067448954.csv')
#df_11 = pd.read_csv(INPUT_PATH+'')
#df_12 = pd.read_csv(INPUT_PATH+'')
#df_13 = pd.read_csv(INPUT_PATH+'')

q = np.array(df_1['is_iceberg'])

#is_iceberg_arr = np.concatenate((np.array(df_1['is_iceberg']), np.array(df_2['is_iceberg']), np.array(df_3['is_iceberg']), np.array(df_4['is_iceberg']), np.array(df_5['is_iceberg']), np.array(df_6['is_iceberg']), np.array(df_7['is_iceberg']), np.array(df_8['is_iceberg']), np.array(df_9['is_iceberg'])))
#is_iceberg_arr = np.concatenate((np.array(df_2['is_iceberg']), np.array(df_3['is_iceberg']), np.array(df_4['is_iceberg'])))
is_iceberg_arr = np.concatenate((np.array(df_1['is_iceberg']), np.array(df_5['is_iceberg']), np.array(df_6['is_iceberg']), np.array(df_7['is_iceberg']), np.array(df_8['is_iceberg']), np.array(df_9['is_iceberg'])))

is_iceberg_arr = is_iceberg_arr.reshape(-1,8424)

print(np.corrcoef(is_iceberg_arr))

is_iceberg_arr = np.transpose(is_iceberg_arr)

is_iceberg_arr = np.mean(is_iceberg_arr,axis = 1)

print(np.corrcoef(is_iceberg_arr,q))

submission = pd.DataFrame({'id': df_1["id"], 'is_iceberg': is_iceberg_arr})
print(submission.head(10))

submission.to_csv(INPUT_PATH + '20171207_submission_ens_28.csv', index=False)


