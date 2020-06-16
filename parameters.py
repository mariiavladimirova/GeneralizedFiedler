#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 02:13:00 2020

@author: dariabystrova
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
        params["Lasso"] = {"alpha" : 0.09}
        params["Ridge"] = {"alpha" : 4.3}
        params["Fiedler"] = {
             'alpha' : 0.3,
             "lr"    : 0.0005,
             "num_iter" :  300,
             "rep"  : 10
             }           
    if data == "Housing":
        params["Lasso"] = {"alpha" : 0.5}
        params["Ridge"] = {"alpha" : 0.12}
        params["Fiedler"] = {
             'alpha' : 0.11,
             "lr"    : 0.0005,
             "num_iter" : 3000,
             "rep"  : 10
             }           
     
    return params       
           
