#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 01:13:21 2020

@author: dariabystrova

Functions to run the experiments
"""

###Breast cancer data
import pyreadr
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, Ridge
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, RepeatedKFold 
from sklearn.metrics import r2_score
from statistics import mean
import fit
import Fiedler_regularization
from fit import model_fit
from tqdm import tqdm
from statistics import mean, stdev
import requests
import pandas as pd
from io import StringIO
from hyper_parameters import opt_parameters
from sklearn.datasets import load_boston
import numpy as np



# function to write logs
def logToFile(filename, result_string):
    file = open(filename, "a+")
    file.write(result_string)
    file.close()


def load_data(dataset):
     scaler = StandardScaler()
     
     if dataset == "bcTCGA":   
        result = pyreadr.read_r('data/X_matrix2.rds') # reading  Rds
        X_bcTCGA = result[None] 
        result2 = pyreadr.read_r('data/Y_matrix.rds') # reading Rds
        y_bcTCGA = result2[None] 
        y_bc=y_bcTCGA['y']
        #Preprocess data
        X_bcTCGA_std = scaler.fit_transform(X_bcTCGA)
        X = X_bcTCGA_std
        y = y_bc
     if dataset == "AutoMpg":   
 
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
        r = requests.get(url)
        file = r.text.replace("\t"," ")
        # list_labels written manually:
        list_labels = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin','car name']
        df = pd.read_csv(StringIO(file),sep="\s+",header = None,names=list_labels)
        df=df[df.horsepower!='?']
        df.horsepower=df.horsepower.astype('float64')
        #features = list(["cylinders", "displacement", "horsepower", "weight", "acceleration",'model year', 'origin'])
        features_num= list(["displacement", "horsepower", "weight", "acceleration"])
        #features_disc=  list(["cylinders",'model year', 'origin'])
        y_auto = df["mpg"].values
        X_auto = df[features_num]     
        X_auto_std= scaler.fit_transform(X_auto)
        X = X_auto_std
        y = y_auto
    
     if dataset == "Housing":
         boston = load_boston()
         boston_df=pd.DataFrame(boston.data,columns=boston.feature_names) 
         boston_df['Price']=boston.target
         boston_num = boston_df.drop(['CHAS','RAD'],axis=1)
         newX=boston_num.drop('Price',axis=1)
         newY=boston_df['Price']
         y_h = newY
         X_house_std= scaler.fit_transform(newX)
         X = X_house_std
         y= y_h
     if dataset == "Simulation":
        np.random.seed(42)
        n = 2000
        p = 20000
        sigma = 1
        X = np.random.normal(loc=0, size=(n,  p))
        beta = np.zeros(p)
        y = np.zeros(n)
        for j in range(24): 
            beta[j] = 2 ** (-j/4 + 9/4)
            
        for i in range(n): 
            y[i] = np.random.normal(loc=np.dot(X[i, :], beta), scale=sigma)
        
     return X,y
           



def run_experiment(data, model):
    X, y = load_data(data)
    parameters = opt_parameters(data)
    scores =  model_fit(model,parameters[model],X,y)
    filename = "logs/{}_{}.txt".format(data,model)
    logToFile(filename, """prediction scores: {}, std: {}\n""".format(scores.mean(),scores.std()))
    
    return scores



def run_simulation_experiment(data):
    X, y = load_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    params= opt_parameters(data)
    linearModel = LinearRegression() 
    linearModel.fit(X_train, y_train) 
    non_zero_linear = np.count_nonzero(linearModel.coef_)
    ridgeModel = Ridge(alpha = params['Ridge']['alpha'],fit_intercept=True, tol=0.00925)
    ridgeModel.fit(X_train, y_train) 
    non_zero_ridge = np.count_nonzero(ridgeModel.coef_)
    lassoModel = Lasso(alpha = params['Lasso']['alpha'],fit_intercept=True, tol=0.00925)
    lassoModel.fit(X_train, y_train) 
    non_zero_lasso = np.count_nonzero(lassoModel.coef_)
    fiedlerModel = Fiedler_regularization.Optimization(lr= params['Fiedler']['lr'], num_iter= params['Fiedler']['num_iter'], reg=Fiedler_regularization.FiedlerRegularization(delta =  params['Fiedler']['alpha']), verbose=True)
    fiedlerModel.fit(X_train, y_train)
    coef_fiedler =  np.delete(fiedlerModel.theta,0)
    non_zero_fiedler = np.count_nonzero(coef_fiedler)
    filename = "logs/{}.txt".format(data)
    models_names = ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'Fiedler Regression'] 
    models = [linearModel, ridgeModel, lassoModel, fiedlerModel] 
    Fiedler_regularization.score_comparison(models_names, models, X_train, y_train, verbose_dataset_type='train', file = filename)
    Fiedler_regularization.score_comparison(models_names, models, X_test, y_test, verbose_dataset_type='test', file = filename)
    Fiedler_regularization.score_comparison(models_names, models, X, y, verbose_dataset_type='whole', file = filename)
    logToFile(filename, """Linear number of non-zero coefficients: {} \n""".format( non_zero_linear))
    logToFile(filename, """Lasso number of non-zero coefficients: {} \n""".format( non_zero_lasso))
    logToFile(filename, """Ridge number of non-zero coefficients: {} \n""".format( non_zero_ridge))
    logToFile(filename, """Fiedler number of non-zero coefficients: {} \n""".format( non_zero_fiedler))
 

def experiment(dataset):
    X, y =  load_data(dataset)
    #X, y =  load_data(dataset)
    params=  opt_parameters(dataset)
    filename = "{}.txt".format(dataset)
    scores,num_non_zero = model_fit("Lasso", params["Lasso"],X,y)
    filename = "logs/{}.txt".format(dataset)
    logToFile(filename, """Lasso prediction scores: {}, std: {}\n""".format(scores.mean(),scores.std()))
    logToFile(filename, """Lasso number of non-zero coefficients: {}, std: {}\n""".format(num_non_zero.mean(),num_non_zero.std()))
    
    scores,num_non_zero = model_fit("Ridge", params["Ridge"],X,y)
    logToFile(filename, """Ridge prediction scores: {}, std: {}\n""".format(scores.mean(),scores.std()))
    logToFile(filename, """Ridge number of non-zero coefficients: {}, std: {}\n""".format(num_non_zero.mean(),num_non_zero.std()))
   
    scores,num_non_zero = model_fit("Fiedler", params["Fiedler"],X,y)
    logToFile(filename, """Fideler prediction scores: {}, std: {}\n""".format(scores.mean(),scores.std()))
    logToFile(filename, """Fiedler number of non-zero coefficients: {}, std: {}\n""".format(num_non_zero.mean(),num_non_zero.std()))
   













