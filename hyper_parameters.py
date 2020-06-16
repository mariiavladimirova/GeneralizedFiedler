#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 02:13:00 2020

@author: dariabystrova

Hyperparameter specification for publication on ICML 2020 workshop
"""

def opt_parameters(data):
    params= {
        "dataset": data
        }
    if data == "bcTCGA":
        params["Lasso"] = {"alpha" : 0.04}
        params["Ridge"] = {"alpha" : 5}
        params["Fiedler"] = {
             'alpha' : 0.09 ,
             "lr"    : 0.0005 ,
             "num_iter" :  3000,
             "rep" : 10
             }           
    if data == "AutoMpg":
        params["Lasso"] = {"alpha" : 0.13}
        params["Ridge"] = {"alpha" : 3.5}
        params["Fiedler"] = {
             'alpha' : 0.113,
             "lr"    : 0.001,
             "num_iter" :  3000,
             "rep"  : 10
             }           
    if data == "Housing":
        params["Lasso"] = {"alpha" : 0.05}
        params["Ridge"] = {"alpha" : 0.14}
        params["Fiedler"] = {
             'alpha' : 0.064,
             "lr"    : 0.001,
             "num_iter" : 5000,
             "rep"  : 10
             }           
    if data == "Simulation":
        params["Lasso"] = {"alpha" : 0.06}
        params["Ridge"] = {"alpha" : 0.1}
        params["Fiedler"] = {
             'alpha' : 0.1,
             "lr"    : 0.01,
             "num_iter" : 1000,
             "rep"  : 1
             }           
         
    return params       
           
