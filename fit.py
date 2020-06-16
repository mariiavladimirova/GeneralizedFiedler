#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 00:38:27 2020
@author: dariabystrova

General fitting function
"""
from sklearn.preprocessing import StandardScaler, add_dummy_feature
import Fiedler_regularization
from Fiedler_regularization import cross_validation_fiedler
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, Ridge
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, RepeatedKFold 
from sklearn.metrics import r2_score
from statistics import mean

import random

import warnings
warnings.filterwarnings('ignore')

#from tqdm import tqdm



def model_fit(model_name, params,X,y):
    if model_name == "Ridge" or model_name == "Lasso":
          if model_name == "Ridge":
              model= Ridge(alpha = params['alpha'],fit_intercept=True, tol=0.00925)
          elif model_name=="Lasso":
               model=Lasso(alpha = params['alpha'],fit_intercept=True, tol=0.00925)
          cv_model = cross_validate(model, X,y,cv=RepeatedKFold(n_splits=10, n_repeats=10),
          return_estimator=True,scoring="neg_mean_squared_error")
          scores= - cv_model['test_score']
          non_zero_coef=[]
          for i in range(100):
              non_zero_coef.append(np.count_nonzero(cv_model['estimator'][i].coef_))
    if model_name=="Fiedler":
         scores,non_zero_coef = cross_validation_fiedler(params['alpha'], X, y,score_name="mse", lr=params['lr'], num_iter=params['num_iter'],replicates=params['rep'])              

    return scores,np.array(non_zero_coef)
    
    

#Ridge regression


