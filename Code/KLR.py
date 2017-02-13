#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 16:11:29 2017

@author: estelleaflalo
"""
import math
import numpy as np
import pandas as pd


X_tr_path='/Users/estelleaflalo/Desktop/M2_Data_Science/Second_Period/Kernel_Methods/Project/Xtr.csv'
Y_tr_path='/Users/estelleaflalo/Desktop/M2_Data_Science/Second_Period/Kernel_Methods/Project/Ytr.csv'

df_X=pd.read_csv(X_tr_path, header=None)
df_X = df_X.iloc[:, :-1]
df_y=pd.read_csv(Y_tr_path, header=None)
#df_y= df_y.iloc[:, :-1]

X=df_X.as_matrix()
y=df_y.as_matrix()[:,1]
y=y[1:].astype(float)


def log_prime(u):
    return -1./(1+np.exp(u))
    
def log_primeprime(u):
    return 1.*np.exp(u)/(1+np.exp(u))**2

class KLR():
    def __init__(self,kernel,alpha_init,lamda):
        self.ker=kernel
        self.alpha0=alpha_init
        self.n=alpha_init.shape[0]
        self.lamb=lamda
        
        
        
    def fit(self, X, y):
        K=self.ker(X)

        temp=(K.dot(self.alpha0))*y
        P=np.diag(log_prime(temp))
        W=np.diag(log_primeprime(temp))
        Z=K.dot(alpha0)-np.linalg.inv(W).dot(P).dot(y)
        alpha=alpha0
        for i in range(10):
            alpha=np.linalg.inv(np.transpose(K).dot(W).dot(K)+self.n*self.lamb*K).dot(np.transpose(Z).dot(W).dot(K))
            m=K.dot(alpha)
            P=np.diag(log_prime(m))
            W=np.diag(log_primeprime(m))
            Z=m-np.linalg.inv(W).dot(P*y)
            J=(1./self.n)*(np.transpose(K.dot(alpha)-Z).dot(W).dot(K.dot(alpha)-Z))+self.lamb*np.transpose(alpha).dot(K).dot(alpha)
            print i, J
        
        
        
    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)


