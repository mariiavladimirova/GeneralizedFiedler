#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 01:55:20 2020

@author: dariabystrova
"""


from experiments import experiment
from experiments import run_simulation_experiment



for i in range(1):
    #Simulation
   # run_simulation_experiment("Simulation")
    
    ##UCI Datasets experiments
    experiment("AutoMpg")
    experiment("Housing")
    
    ##GENE
    #experiment("bcTCGA")
 