#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 11:41:27 2017

@author: estelleaflalo
"""

import pandas as pd
import numpy as np
from accuracy import accuracy_score
import KLR_algo
import SVM_algo
import kernel_functions
import HOG
#estelle
#X_tr_path='/Users/estelleaflalo/Desktop/M2_Data_Science/Second_Period/Kernel_Methods/Project/Xtr.csv'
#Y_tr_path='/Users/estelleaflalo/Desktop/M2_Data_Science/Second_Period/Kernel_Methods/Project/Ytr.csv'

#Pauline
X_tr_path = '/Users/paulinenicolas/Documents/M2_Data_Science/Kernels/Project/Xtr.csv'
Y_tr_path = '/Users/paulinenicolas/Documents/M2_Data_Science/Kernels/Project/Ytr.csv'

df_X=pd.read_csv(X_tr_path, header=None)
df_X = df_X.iloc[:, :-1]
df_y=pd.read_csv(Y_tr_path, header=None)
#df_y= df_y.iloc[:, :-1]

X=df_X.as_matrix()
y=df_y.as_matrix()[:,1]
y=y[1:]
y=y.astype(float)



#Cross validation
#splitting the dataset in a train and test set
idx_info = np.arange(X.shape[0])
np.random.shuffle(idx_info)
idx_train = idx_info[:int(0.8*X.shape[0])]
idx_test = idx_info[int(0.8*X.shape[0]):]

X_train, X_test, y_train, y_test = X[idx_train], X[idx_test], y[idx_train], y[idx_test]

print("Nb d'échantillons d'apprentissage :  {}".format(X_train.shape[0]))
print("Nb d'échantillons de validation :    {}".format(X_test.shape[0]))



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
print ("The parameters of the HOG model are the following : number of bins for the orientations histograms = %d, number of blocks : %d, number of cells per block : %d"%(nbins,nblocks,ncells))

temp1=np.zeros(9*ncells*nbins) #mean #nbins (ici 9) * nb_position_blocks (ici 9) * nombre de cellules par block (4)
#temp1=np.zeros(9*ncells*nbins*3)  #all channels
for i in range(X_train.shape[0]):
    image=X_train[i]
    gx,gy=HOG.gradients(image)
    allhisto=HOG.all_histo(image,gx,gy,nbins)
    temp1=np.vstack((temp1,HOG.hog_vector(allhisto)))
    if i%100==0:
        print (i)
        
new_features_train=temp1[1:]     
#new_features_train=np.sqrt(new_features_train)

temp2=np.zeros(9*ncells*nbins) #mean #nbins (ici 9) * nb_position_blocks (ici 9) * nombre de cellules par block (4)
#temp2=np.zeros(9*ncells*nbins*3)  #all channels
for i in range(X_test.shape[0]):
    image=X_test[i]
    gx,gy=HOG.gradients(image)
    allhisto=HOG.all_histo(image,gx,gy,nbins)
    temp2=np.vstack((temp2,HOG.hog_vector(allhisto)))
    if i%100==0:
        print (i)
        
new_features_test=temp2[1:]       
#new_features_test=np.sqrt(new_features_test)
#SVM
print ("Building the SVM-multiclass model")


kernel=kernel_functions.gaussian_kernel
kernel_name="gaussian kernel"
print ("We used the %s for the SVM"%(kernel_name))


X_train=new_features_train
X_test=new_features_test


#SVM
total_pred = np.zeros(len(y_test))

K_Train=kernel_functions.gaussian_kernel(X_train,X_train)
for i in range(10):
    classe = i
    model = SVM_algo.SVM(0.5,K_Train,kernel_functions.gaussian_kernel,.1, classe)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(model.score(y_test))
    total_pred = np.hstack((total_pred,y_pred))

total_pred = total_pred.reshape((11,1000)).T
total_pred = total_pred[:,1:]
final_pred = np.argmax(total_pred, axis=1)


#accuracy_score(np.sign(self.y_pred), y_te)

y_pred=final_pred

# accuracy : pourcentage de bonnes predictions
print("Accuracy       : ", accuracy_score(y_test, y_pred))
print("Accuracy bis:  : ", np.mean(y_test == y_pred)) # mesure d'erreur 0/1

print("Le classifieur propose une bonne prédiction dans {} % des cas.".format(
      100 * accuracy_score(y_test, y_pred))) 

#
##KLR
#
#for i in range(len(y_train)):
#    if y_train[i]==0:
#        y_train[i]=-1
#    else:
#        y_train[i]=1
#
#for i in range(len(y_test)):
#    if y_test[i]==0:
#        y_test[i]=-1
#    else:
#        y_test[i]=1
#
#n=X_train.shape[0]
#alpha0=np.zeros(n)
#model = KLR_algo.KLR(alpha0,1, kernel_test)
#alpha,J_list,alpha_list = model.fit(X_train,y_train)
#
#
##prediction test
#y_pred = []
#
#for i in range(X_test.shape[0]):
#    pred = 0
#    for j in  range(X_train.shape[0]):
#        pred += alpha[j]*X_train[j].T.dot(X_test[i])
#    y_pred.append(pred)
#    if i%100==0:
#        print (i)
# 
#for i in range(len(y_pred)):
#    if y_pred[i]>0:
#        y_pred[i]=1
#    else:
#        y_pred[i]=-1
##y_pred=model.predict(X_test)
#
#n=X_train.shape[0]
#alpha0=np.zeros(n)
#model=KLR_algo.KLR(alpha0,0.5,kernel_test)
#model.fit(X_train,y_train)
#y_pred=model.predict(X_test)
#
## accuracy : pourcentage de bonnes predictions
#print("Accuracy       : ", accuracy_score(y_test, y_pred))
#print("Accuracy bis:  : ", np.mean(y_test == y_pred)) # mesure d'erreur 0/1
#
#print("Le classifieur propose une bonne prédiction dans {} % des cas.".format(
#      100 * accuracy_score(y_test, y_pred))) 
#
