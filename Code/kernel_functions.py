#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 16:04:16 2017

@author: paulinenicolas
"""

import numpy as np

#Linear Kernel
def linear_kernel(x, y, c):
    return np.dot(x.T, y) + c

#Polynomial kernel
def polynomial_kernel(x, y, a, b, c, d):
    return (a * np.dot(x.T, y) + c)**d


#Gaussian kernel
def gaussian_kernel(x, y, gamma):
    return np.exp(-gamma*np.linalg.norm(x-y, ord=2)**2)

#Laplacian Kernel
def laplacian_kernel(x, y, sigma):
    return np.exp(-np.linalg.norm(x-y, ord=2)/sigma)
    
def kernel_test(X,Y):
    return X.dot(np.transpose(Y))
