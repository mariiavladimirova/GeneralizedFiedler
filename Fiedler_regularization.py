# -*- coding: utf-8 -*-
"""
Functions computing fiedler value and derivatives.
"""

import numpy as np
#import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, Ridge
from sklearn.model_selection import train_test_split, cross_val_score 
from sklearn.metrics import r2_score
from statistics import mean
from sklearn.model_selection import KFold
import random
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
np.random.seed(42)


def logToFile(filename, result_string):
    file = open(filename, "a+")
    file.write(result_string)
    file.close()





def fiedler_value(w):
    return np.linalg.norm(w, ord=1) \
            - np.sqrt(
                    np.linalg.norm(w, ord=2) ** 2 - ( \
                        np.linalg.norm(w, ord=1) ** 2 \
                        - np.linalg.norm(w, ord=2) ** 2
                        ) / (len(w) - 1) + 5e-15
                    )
                
                
                
def fiedler_derivative(w):
    p = len(w) 
    w = np.array(w)
    denominator = np.sqrt( np.linalg.norm(w, ord=2) ** 2 - (
                                                     np.linalg.norm(w, ord=1) ** 2 \
                                                    - np.linalg.norm(w, ord=2) ** 2 ) / (p - 1)
                         )
    
    if denominator != 0:
        res = np.sign(w) - (w * p - np.sum(np.abs(w)) * np.sign(w)) / (p - 1) / denominator
    else:
        res = [1] * p
                                 
    return res


def fiedler_second_derivative(w):
    p = len(w) 
    w = np.array(w)
    denominator = np.sqrt( np.linalg.norm(w, ord=2) ** 2 - (
                                                    np.linalg.norm(w, ord=1) ** 2 \
                                                    - np.linalg.norm(w, ord=2) ** 2 ) / (p - 1)
                         )
    
    if denominator != 0:
        common = [ - 1 / denominator + 3 / (p - 1) ** 2 * ( - np.linalg.norm(w, ord=1) ** 2 * (p - 1) \
                                    + np.linalg.norm(w, ord=2) ** 2 * p ** 2) / denominator ** 3
             ] * p
    
        rest = np.power(w, 2) * 3 * p ** 2 / (p - 1) ** 2 / denominator ** 3
        res = common + rest 
    else: 
        res = [1] * p
                                     
    return res


def g_prox(x,gamma, delta):
    dim =np.size(x) 
    x_new = np.zeros(dim)
    for i in range(dim):
        if x[i] < - delta[i]*gamma:
            x_new[i] = x[i] + delta[i]*gamma
        if x[i] > delta[i]*gamma:
            x_new[i] = x[i] - delta[i]*gamma
    return x_new

def mse(y, y_pred):
    return np.sum((y - y_pred) ** 2) / len(y)

def score_comparison(models_names, models, X, y, metric='R2', verbose_dataset_type=None,file = None):
    # Building the dictionary to compare the scores 
    mapping = {} 
    for i in range(len(models)):
        if metric == 'R2':
            mapping[models_names[i]] = models[i].score(X, y)
        if metric == 'MSE':
            mapping[models_names[i]] = mse(y, models[i].predict(X))
                
    if file is None:       
            if verbose_dataset_type:
                print(metric, "on a " + verbose_dataset_type + " set")
            # Printing the scores for different models 
            for key, val in mapping.items(): 
                print(str(key)+' : '+str(val)) 
    else:      
        for key, val in mapping.items(): 
          logToFile(file, """{} on a {} set """.format(metric,verbose_dataset_type))  
          logToFile(file, """{} :  {}  \n""".format(key,val))  
        logToFile(file, """--------------------------\n""")       
                
                    

###### Regularization with Fiedler

class Regularization: 
    def __init__(self, delta = 0.5):
        self.delta = delta
    
    def penalty(self, w):
        return 0
    
    def gradient(self, w):
        return 0
    
class L1(Regularization):
    def __init__(self, delta = 0.5):
        super().__init__(delta)  

    def penalty(self, w):
        return np.linalg.norm(w, ord=1)
    
    def gradient(self, w):
        return np.sign(w)  

class L2(Regularization):
    def __init__(self, delta = 0.5):
        super().__init__(delta) 
        
    def penalty(self, w):
        return np.linalg.norm(w, ord=2)
    
    def gradient(self, w):
        return 2 * w
    
class FiedlerRegularization(Regularization):
    def __init__(self, delta = 0.5):
        super().__init__(delta) 
        
    def penalty(self, w):
        return np.sum(np.multiply(fiedler_derivative(np.abs(w)), np.abs(w)))
    
    def gradient(self, w):
        if np.all(w == np.zeros(len(w))):
            res = np.zeros(len(w))
        else:
            res = np.multiply(fiedler_derivative(np.abs(w)), np.sign(w))    
        return res


class Optimization:
    def __init__(self, lr=0.01, num_iter=100000, reg=None, tol=1e-4, loss_function="squared", 
                        fit_intercept=False, alg_prox =True,verbose=False):
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.reg = Regularization() if reg == None else reg
        self.tol = tol
        self.loss_function = loss_function
        self.theta = None
        self.alg_prox =alg_prox 
    
    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def __loss(self, X, y, theta):
        if self.loss_function == "squared": 
            f = np.sum((y - np.dot(X, theta)) ** 2 / y.size)
        return f
    
    def __loss_gradient(self, X, y, theta):
        if self.loss_function == "squared": 
            f_gradient = 2 * np.dot(X.T, (np.dot(X, theta) - y)) / y.size
        return f_gradient
    
    
    def fit(self, X, y):
        if self.fit_intercept:
            X = self.__add_intercept(X)
        
        # weights initialization
        self.theta = np.zeros(X.shape[1])
        n_iter = 0
        gradient = self.tol * 10
        #x_tab= np.copy(self.theta)
       # gradient = self.tol * np.linalg.norm(self.__loss_gradient(X, y, self.theta))
       # f_tab=self.reg.penalty(self.theta) + self.__loss(X,y,self.theta)
        while n_iter <= self.num_iter and np.any(gradient > self.tol):
            if not self.alg_prox:
                gradient = self.__loss_gradient(X, y, self.theta) + self.reg.delta * self.reg.gradient(self.theta)
                self.theta -= self.lr * gradient
                
            if self.alg_prox:
                gradient = self.__loss_gradient(X, y, self.theta)
                self.theta = g_prox(self.theta -self.lr*gradient,self.lr,(np.multiply(self.reg.delta,fiedler_derivative(np.abs(self.theta)))))  
                # x = g_prox(x - step*g , step)  
            n_iter += 1
            #x_tab= np.vstack((x_tab,self.theta))
            #f_tab = np.vstack((f_tab,(self.reg.penalty(self.theta) + self.__loss(X,y,self.theta))))
            if(self.verbose == True and n_iter % 500 == 0):
                print(f'loss: {self.__loss(X, y, self.theta)}')
           # if(self.verbose == True and n_iter % 500 == 0):
           #     print(f'f_approx: {(self.__loss(X, y, self.theta) + self.reg.penalty(self.theta))}')
           # if(self.verbose == True and n_iter % 500 == 0):
           #     print(f'f_delta: {(f_tab[-1] - f_tab[-2])}')
            
           # if(self.verbose == True and n_iter % 500 == 0):
           #     print(f'grad_norm: {(np.linalg.norm(gradient))}')
                
        
        print('Iterations were done:', n_iter-1)
#         print(gradient)
#        return x_tab,f_tab 
    
    def predict(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)
    
        return np.dot(X, self.theta)
    
    def score(self, X, y, sample_weight=None, name='r2'):
        if name=='r2':
            res = r2_score(y, self.predict(X), sample_weight=sample_weight)
            
        if name=='mse':
            res = np.sum((y - self.predict(X)) ** 2) / len(y)

        return res
    
    
    
    
 #Cross-validation   

def cross_validation_fiedler(delta_val,X,y, score_name="r2", prox=True,lr=0.01, num_iter=1000, replicates=1):
    scores = []
    print(delta_val)
    rand_state = np.linspace(1, 100, replicates)
    non_zero =[]
    for i in tqdm(range(0,replicates)): 
        cv = KFold(n_splits=10, random_state=int(rand_state[i]), shuffle=True)
        for train_index_, test_index_ in cv.split(X):
            X_train_, X_test_, y_train_, y_test_ = X[train_index_], X[test_index_], y[train_index_], y[test_index_]
            fiedlerModel =  Optimization(lr=lr, num_iter=num_iter, fit_intercept=True, alg_prox=prox, reg=FiedlerRegularization(delta=delta_val), verbose=True)
            fiedlerModel.fit(X_train_, y_train_) 
            scores.append(fiedlerModel.score(X_test_, y_test_, name=score_name))
            coef_fiedler =  np.delete(fiedlerModel.theta,0)
            non_zero.append(np.count_nonzero(coef_fiedler))    
    return np.array(scores),np.array(non_zero)











                