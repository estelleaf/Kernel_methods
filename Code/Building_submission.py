#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 12:51:07 2017

@author: estelleaflalo
"""
import SVM_algo
import kernel_functions
import pandas as pd
import HOG
import numpy as np
#estelle
X_tr_path='/Users/estelleaflalo/Desktop/M2_Data_Science/Second_Period/Kernel_Methods/Project/Xtr.csv'
Y_tr_path='/Users/estelleaflalo/Desktop/M2_Data_Science/Second_Period/Kernel_Methods/Project/Ytr.csv'
X_te_path='/Users/estelleaflalo/Desktop/M2_Data_Science/Second_Period/Kernel_Methods/Project/Xte.csv'

print "Loading the dataframes"
df_X=pd.read_csv(X_tr_path, header=None)
df_X = df_X.iloc[:, :-1]
df_y=pd.read_csv(Y_tr_path, header=None)
#df_y= df_y.iloc[:, :-1]
X_train=df_X.as_matrix()
y_tr=df_y.as_matrix()[:,1]
y_tr=y_tr[1:]
y_train=y_tr.astype(float)

df_X=pd.read_csv(X_te_path, header=None)
df_X = df_X.iloc[:, :-1]
X_test=df_X.as_matrix()


print("Size of training set :  {}".format(X_train.shape[0]))
print("Size of testing set :    {}".format(X_test.shape[0]))




print "Preprocessing of the images"

Xtr_reshape = X_train.reshape(( X_train.shape[0],3,32,32)).transpose(0,2,3,1)
for i in range(len(Xtr_reshape)):
    Xtr_reshape[i,:,:,0] -= Xtr_reshape[i,:,:,0].min()
    Xtr_reshape[i,:,:,1] -= Xtr_reshape[i,:,:,1].min()
    Xtr_reshape[i,:,:,2] -= Xtr_reshape[i,:,:,2].min()
X_train=Xtr_reshape   
 
Xte_reshape = X_test.reshape((X_test.shape[0],3,32,32)).transpose(0,2,3,1)
for i in range(len(Xte_reshape)):
    Xte_reshape[i,:,:,0] -= Xte_reshape[i,:,:,0].min()
    Xte_reshape[i,:,:,1] -= Xte_reshape[i,:,:,1].min()
    Xte_reshape[i,:,:,2] -= Xte_reshape[i,:,:,2].min()
X_test=Xte_reshape    

print ("Building the features matrices (train and test) based on HOG model")
nbins=9
nblocks=4
ncells=4
print "The parameters of the HOG model are the following : number of bins for the orientations histograms = %d, number of blocks : %d, number of cells per block : %d"%(nbins,nblocks,ncells)

temp1=np.zeros(324) #nbins (ici 9) * nb_position_blocks (ici 9) * nombre de cellules par block (4)
for i in range(X_train.shape[0]):
    image=X_train[i]
    gx,gy=HOG.gradients(image)
    allhisto=HOG.all_histo(image,gx,gy,nbins)
    temp1=np.vstack((temp1,HOG.hog_vector(allhisto)))
    if i%100==0:
        print i
        
new_features_train=temp1[1:]     


temp2=np.zeros(324) #nbins (ici 9) * nb_position_blocks (ici 9) * nombre de cellules par block (4)
for i in range(X_test.shape[0]):
    image=X_test[i]
    gx,gy=HOG.gradients(image)
    allhisto=HOG.all_histo(image,gx,gy,nbins)
    temp2=np.vstack((temp2,HOG.hog_vector(allhisto)))
    if i%100==0:
        print i
        
new_features_test=temp2[1:]       


#SVM
print "Building the SVM-multiclass model"


kernel=kernel_functions.kernel_test
kernel_name="kernel_functions.kernel_test"
print ("We used the %s for the SVM"%(kernel_name))

Ktrain = kernel(X_train, X_train)
total_pred = np.zeros(len(X_test[0]))
for i in range(10):
    classe = i
    model = SVM_algo.SVM(0.5,Ktrain,kernel,.1, classe)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    total_pred = np.hstack((total_pred,y_pred))

total_pred = total_pred.reshape((11,1000)).T
total_pred = total_pred[:,1:]
final_pred = np.argmax(total_pred, axis=1)

