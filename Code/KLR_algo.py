#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 16:11:29 2017

@author: estelleaflalo
"""
import math
import numpy as np
import pandas as pd
import kernel_functions

def logistic(u):
    return np.log(1./(1+np.exp(-u)))

def log_prime(u):
    return -1./(1+np.exp(u))
    
def log_primeprime(u):
    return 1.*np.exp(u)/(1+np.exp(u))**2

class KLR():
    def __init__(self,alpha_init,lamda):#,kernel):
        #self.ker=kernel
        self.alpha0=alpha_init
        self.n=alpha_init.shape[0]
        self.lamb=lamda
        
        
        
    def fit(self, X, y):
        #K=self.ker(X)
        alpha_list=[]
        J_list=[]
        K=X.dot(np.transpose(X))
        temp=(K.dot(self.alpha0))*y
        P=np.diag(log_prime(temp))
        W=np.diag(log_primeprime(temp))
        Z=K.dot(alpha0)-np.linalg.inv(W).dot(P).dot(y)
        alpha_list.append(alpha0)
        for i in range(15):
            alpha=np.linalg.inv(np.transpose(K).dot(W).dot(K)+self.n*self.lamb*K).dot(np.transpose(Z).dot(W).dot(K))
            m=K.dot(alpha)
            P=np.diag(log_prime(m*y))
            #print P
            W=np.diag(log_primeprime(m*y))
            #print W
            Z=m-np.linalg.inv(W).dot(P).dot(y)
            #J=(1./self.n)*(np.transpose(K.dot(alpha)-Z).dot(W).dot(K.dot(alpha)-Z))+self.lamb*np.transpose(alpha).dot(K).dot(alpha)
            J=(1./self.n)*sum(logistic(y*m))+(self.lamb/2.)*np.transpose(alpha).dot(K).dot(alpha)
            alpha_list.append(alpha)
            J_list.append(J)
            print i, J
        return alpha,J_list,alpha_list
        
        
        
    def fit_predict(self, Xtr,Ytr,Xte):
        K=self.ker(Xte)
        return K.dot(self.fit(Xtr,Ytr))
        
        

    def predict_proba(self, X):
        return self.clf.predict_proba(X)
        
n=X.shape[0]
alpha0=np.zeros(n)
model=KLR(alpha0,0.5)
model.fit(X,y)
