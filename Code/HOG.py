#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 11:41:59 2017

@author: domitillecoulomb
"""


import pandas as pd
from numpy.linalg import norm
import numpy as np

"""
Gradients
"""

def gradients(image):
    
    gx = np.zeros(image.shape)
    gy = np.zeros(image.shape)
    
    for col in range(image.shape[2]):
        gx[:, 1:-1, col] = -image[:, :-2, col] + image[:, 2:, col]
        gx[:, 0, col] = -image[:, 0, col] + image[:, 1, col]
        gx[:, -1, col] = -image[:, -2, col] + image[:, -1, col]
        
        gy[1:-1, :, col] = image[:-2, :, col] - image[2:, :, col]
        gy[0, :, col] = image[0, :, col] - image[1, :, col]
        gy[-1, :, col] = image[-2, :, col] - image[-1, :, col]
    
    return gx, gy

def magnitude_orientation(gx, gy):
    
    """ 
    Computes the magnitude and orientation matrices from the gradients gx gy
    The orientation is in degree, NOT radian!!
    """
        
    magnitude = np.sqrt(gx**2 + gy**2)
    orientation = (np.arctan2(gy, gx) * 180 / np.pi) % 180
                  
    return magnitude, orientation

def cell_histogram(x0,y0,gx,gy,nbins):
    
    magnitude, orientation = magnitude_orientation(gx[y0:y0+8, x0:x0+8,:], gy[y0:y0+8, x0:x0+8,:])
    
    #Box step
    b_step = 180/nbins
    
    #Find the boxes
    b0 = np.floor(orientation // b_step)
    b0[np.where(b0>=nbins)]=0
    b1 = b0 + 1
    b1[np.where(b1>=nbins)]=0
    
    # ratio de distance aux 2 box les plus proches
    ratio = np.abs(orientation % b_step).astype(float) / float(b_step)
    
    #Compute histogram of cell
    histogram = np.zeros((nbins,3))
    
    m0 = magnitude * (1 - ratio)
    m1 = magnitude * ratio
    
    for i in range(nbins):
        for c in range(3): #channels
            histogram[i,c] += np.sum(m0[:,:,c][np.where(b0[:,:,c]==i)])
            histogram[i,c] += np.sum(m1[:,:,c][np.where(b1[:,:,c]==i)])
    
    #return np.mean(histogram,axis=1) #mean
    return np.hstack(histogram.T) #all channels

def all_histo(image,gx,gy,nbins):
    #all_histo=np.zeros((4,4,nbins)) #with mean
    all_histo=np.zeros((4,4,nbins*3)) #with all channels
    for i in range(4):
        for j in range(4):
            all_histo[i,j,:]=cell_histogram(8*i,8*j,gx,gy,nbins)
    return all_histo

def hog_vector(allhisto): 
    temp_old=np.zeros(1)
    for i in range(3):
        for j in range(3):
            temp=np.concatenate((allhisto[i,j,:],allhisto[i+1,j,:],allhisto[i,j+1,:],allhisto[i+1,j+1,:]),axis=0)
            temp=temp.astype(float)/norm(temp,ord=2)
            temp=np.concatenate((temp_old,temp))
            temp_old=temp
    temp=temp[1:]
    return temp
    
'''
#X_tr_path = '/Users/paulinenicolas/Documents/M2_Data_Science/Kernels/Project/Xtr.csv'
#Y_tr_path = '/Users/paulinenicolas/Documents/M2_Data_Science/Kernels/Project/Ytr.csv'
X_tr_path = '/Users/domitillecoulomb/M2_DataSciences/Kernels/Xtr.csv'
Y_tr_path = '/Users/domitillecoulomb/M2_DataSciences/Kernels/Ytr.csv'

df_X=pd.read_csv(X_tr_path, header=None)
df_X = df_X.iloc[:, :-1]
df_y=pd.read_csv(Y_tr_path, header=None)
#df_y= df_y.iloc[:, :-1]

X=df_X.as_matrix()
y=df_y.as_matrix()[:,1]
y=y[1:]
y=y.astype(float)

"""
Preprocessing
"""
X_reshape = X.reshape((5000,3,32,32)).transpose(0,2,3,1)
for i in range(len(X_reshape)):
    X_reshape[i,:,:,0] -= X_reshape[i,:,:,0].min()
    X_reshape[i,:,:,1] -= X_reshape[i,:,:,1].min()
    X_reshape[i,:,:,2] -= X_reshape[i,:,:,2].min()


nbins=9
old=np.zeros(324) #nbins (ici 9) * nb_position_blocks (ici 9) * nombre de cellules par block (4)
for i in range(X_reshape.shape[0]):
    image=X_reshape[i]
    gx,gy=gradients(image)
    allhisto=all_histo(image,gx,gy,nbins)
    temp=np.vstack((old,hog_vector(allhisto)))
    old=temp
    if i%100==0:
        print i
        
new_features=temp[1:]       


    

"""
10 patches per image
"""
"""
patches = []
for im in range (len(X_reshape)):
    for i in range(8):
        for j in range(8):
            patches.append( X_reshape[im,4*i:4*(i+1),4*j:4*(j+1)] )"""

from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC            
X_train, X_test, y_train, y_test = train_test_split(new_features, y, train_size=0.8,
                                                    random_state=52)
from sklearn.metrics import accuracy_score

model=SVC(C=1,kernel='linear')
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
accuracy_score(y_test, y_pred)
'''