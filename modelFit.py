#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 17:28:14 2020

@author: vman
"""

import numpy as np
import pandas as pd
np.seterr(all='warn')
import warnings
# Numpy options
import os
cwd = os.path.dirname(os.path.realpath(__file__))
os.chdir(cwd)
from optimizer import *
from utilities import *
from Models import *

    
def fitModel_sess():
    # Initialize paths and sublist
    initDict = initialize() # Define custom initiliazation function for study
    
    # Define list of models to test    
    model_list = {'HMM_dj': ModelHMM}
    
    # Initialize results container
    outFile = f'{initDict.outDir}/{initDict.species}_model_results.csv'
    modCompareDF = pd.DataFrame() 
    # Check if there are interim resuts: 
    if os.path.exists(outFile):
        modCompareDF = pd.read_csv(outFile)
    
    # Iterate over subjects and sessions
    for subIdx, subID in enumerate(initDict.subList):
        # Load in data
        inFile = f'{subID}_data.csv'
        subDF = pd.read_csv(f'{initDict.datDir}/{inFile}')
        sessList = subDF.sessID.unique()

        for sessIdx, sess in enumerate(sessList):
            print(f'Fitting subject: {subID} - session: {sess}')
            sessDict = dict(subID=subID, sess=sess)
            sessDF = subDF[subDF.sessID == sess]
            # Unpack the task sessions
            taskData = unpackTask(sessDF, initDict)
            # Initialize the optimizer
            modOptimizer = Optimizer()
            
            # Check if the model has been run already        
            finished_models = [] 
            if not modCompareDF.empty: 
                finished_models = modCompareDF.model[(modCompareDF['subID'] == subID) & (
                    modCompareDF['sess'] == sess)].unique()
                
            # Iterate over models 
            for model_label, Model in model_list.items():                                    
                if model_label not in finished_models:                 
                    print(f'Model {model_label}')            
                    mod_res, mod_estParams = modOptimizer.fitModel(Model, taskData)
                    # Update results container with estimated model parameters 
                    getattr(mod_res, model_label)(mod_estParams)                    
                    # Append results for model comparison
                    with warnings.catch_warnings():
                        warnings.simplefilter(action='ignore', category=FutureWarning)
                        modCompareDF = modCompareDF.append(dict(sessDict,
                                                                **dict({'model': model_label},
                                                                    **mod_res.__dict__)),
                                                        ignore_index=True)
                        # Output to dataframe
                        modCompareDF.to_csv(outFile, index=False)
                                     
    modCompareDF.to_csv(f'{initDict.outDir}/{initDict.species}_model_results.csv',
                        index=False)
    return

if __name__ == "__main__":
    fitModel_sess()
