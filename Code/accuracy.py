#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 14:35:06 2017

@author: estelleaflalo
"""
import numpy as np

def accuracy_score(y_true, y_pred):
    temp=y_true[y_pred==y_true]
    return float(temp.shape[0])/y_pred.shape[0]


#TEST

#y_pred = [0, 2, 1, 3]
#y_true = [0, 1, 2, 3]

#y_pred=np.asarray(y_pred)
#y_true=np.asarray(y_true)