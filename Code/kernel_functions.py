#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 16:04:16 2017

@author: paulinenicolas
"""

import numpy as np

#Linear Kernel
def linear_kernel(X,Y, c):
    return np.dot(X, Y.T)

#Polynomial kernel
def polynomial_kernel(X, Y, a, b, c, d):
    
    K = np.zeros((X.shape[0], Y.shape[0]))
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            K[i,j] = (a * np.dot(X[i].T, Y[j]) + c)**d
            print (i)
    return K


#Gaussian kernel
def gaussian_kernel(X, Y, gamma):
    K = np.zeros((X.shape[0], Y.shape[0]))
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            K[i,j] = np.exp(-gamma*np.linalg.norm(X[i]-Y[j], ord=2)**2)
            print (i)
    return K

#Laplacian Kernel
def laplacian_kernel(X, Y, sigma):
    K = np.zeros((X.shape[0], Y.shape[0]))
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            K[i,j]= np.exp(-np.linalg.norm(X[i]-Y[j], ord=2)/sigma)
        print (i)
    return K
    
def kernel_test(X,Y):
    return X.dot(np.transpose(Y))
