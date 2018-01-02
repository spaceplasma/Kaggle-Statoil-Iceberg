# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 09:45:28 2017

@author: bryacha
"""
import pandas as pd 
import numpy as np 
from matplotlib import pyplot as plt
from keras.layers import Input, MaxPooling2D, Conv2D, UpSampling2D, ZeroPadding2D
#, Reshape, Cropping2D, Flatten, Lambda, Dense
from keras.models import Model
from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance


#%matplotlib inline

INPUT_PATH='C:\\Users\\bryacha\\projects\\Kaggle\\Iceberg\\data\\processed\\' 

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
#    R = []
    
    for i, row in df.iterrows():
        #make 75x75 image
        band_1 = np.array(row['band_1']).reshape(75, 75)
        band_2 = np.array(row['band_2']).reshape(75, 75)
        band_3 = band_1 + band_2 # plus since log(x*y) = log(x) + log(y)
        
        band_1 = lee_filter(band_1,5)
        
        # Rescale
        a = (band_1 - band_1.mean()) / (band_1.max() - band_1.min())
        b = (band_2 - band_2.mean()) / (band_2.max() - band_2.min())
        c = (band_3 - band_3.mean()) / (band_3.max() - band_3.min())
        
        
#        af = lee_filter(a,5)
#        bf = lee_filter(b,3)
#        cf = lee_filter(c,3)

        imgs.append(np.dstack((a, b, c)))

    return np.array(imgs)

# test denoising utoencoder on training data

print('Fetching Training Data...') 
df_train = pd.read_json(INPUT_PATH + 'train.json')    
    
#df_train = pd.DataFrame(data)

df_train.inc_angle = df_train.inc_angle.replace('na',0)
idx_tr = np.where(df_train.inc_angle>0)
Ytrain = np.array(df_train['is_iceberg'])
Xtrain = get_scaled_imgs(df_train)

### ---
def dae():
    # this is our input placeholder
    input_img = Input(shape=(75, 75, 3))

    # "encoded" is the encoded representation of the input
    x = Conv2D(64, (2, 2), activation='relu', data_format='channels_last', padding="same")(input_img)
    x = MaxPooling2D((3, 3), data_format='channels_last', padding="same")(x) # need 3x3 to divide nicely
    encoded = x
    
    x = Conv2D(64, (2, 2), activation='relu', data_format='channels_last', padding="same")(encoded)
    x = UpSampling2D((3, 3), data_format='channels_last')(x)
    decoded = Conv2D(3, (1, 1), activation='sigmoid', data_format='channels_last', padding="same")(x)
    
    # this model maps an input to its reconstruction
    autoencoder = Model(input_img, decoded)
    
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    
    return autoencoder

#autoencoder = dae()
#autoencoder.summary()
#
#autoencoder.fit(Xtrain, Xtrain, epochs=5, batch_size=32, shuffle=True, validation_split=0.25)
#
#denoised_data = autoencoder.predict(Xtrain)

### ---


band_1 = np.array(df_train['band_1'][0]).reshape(75, 75)
a = (band_1 - band_1.mean()) / (band_1.max() - band_1.min())
imgplot = plt.imshow(a)
plt.show()

imgplot = plt.imshow(Xtrain[0,:,:,0])
plt.show()



