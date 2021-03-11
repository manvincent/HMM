#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 09:56:26 2019

@author: vman
"""

## Make sure to activate the Intel Python distribution
## (bash): source activate IDP

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import sys
# Numpy options
np.seterr(over='warn')
np.set_printoptions(threshold=sys.maxsize)
import pickle
import os
os.chdir('/path/to/project')
# os.chdir('/home/vman/Dropbox/PostDoctoral/Projects/rew_mod/analysis/modelling/HMM/discrete_condaction_HMM')

from optimizer import *
from utilities import *

### Recovery script

def runRecover():
    numMaxDays = 1
    sessPerDays = 5
    fitModel(numMaxDays, sessPerDays)


def initRecover(numSessions):
    modelName = 'cond_action'
    emission_type = 'discrete'
    homeDir = '/path/to/project'
    datDir =  f'{homeDir}/Generate_{modelName}_{emission_type}/Sessions_{numSessions}'
    outDir = f'{homeDir}/Recover_{modelName}_{emission_type}'
    if not os.path.exists(outDir):
        os.makedirs(outDir)
    # Initialize the Model class
    initDict = dict2class(dict(modelName = modelName,
                               emission_type = emission_type,
                               homeDir = homeDir,
                               datDir = datDir,
                               outDir = outDir))
    return(initDict)


def fitModel(numMaxDays, sessPerDays):
    # Loop through different 'experiments' of varying session numbers
    recovCorrDF = pd.DataFrame()
    for sampleIdx, numSessions in enumerate(np.arange(1,numMaxDays+1) * sessPerDays):
        # Print current sample size
        print(f'Recovery with {numSessions} sessions')
        # Initialize the model fitting object
        initDict = initRecover(numSessions)
        # Count number of sims
        _, _, files = next(os.walk(initDict.datDir))
        numIter = len(files)
        # Create structures for storing estimates across iterations (subjects)
        sampleDF = pd.DataFrame()
        # Loop through iterations
        for iterID in np.arange(numIter):
            # Load procedure
            inFile = files[iterID]
            # Load in data
            dataFile = load_obj(f'{initDict.datDir}/{inFile}')
            genModel = dataFile['Model']
            genTask = dataFile['Task']
            # Unpack the task sessions
            taskData = unpackTask(genTask)
            numTrials = len(taskData.highChosen)
            # Print current fit details
            print(f'Iteration No: {iterID+1} out of {numIter}')
            # Initialize the optimizer
            initOptimizer = Optimizer()
            # Run the optimizer
            parallelResults = initOptimizer.getFit(taskData, numTrials, initDict.emission_type)
            # Store estimates from this seed iteration
            fitParams = np.array([parallelResults[s]['x'] for s in np.arange(len(parallelResults))])
            fitNLL = np.array([parallelResults[s]['fun'] for s in np.arange(len(parallelResults))])
            # Store best fitted parameter likelihoods and values
            minIdx = np.argmin(fitNLL)
            recovParams = fitParamContain(fitNLL[minIdx]).action_HMM_fitParams(fitParams[minIdx])
            # Store generative parameter values
            genParams = genModel.genParams
            # Append to dataframe
            sampleDF = sampleDF.append({'gen_delta':genParams['delta'],
                                        'recov_delta':recovParams.delta,
                                        'gen_smBeta':genParams['smBeta'],
                                        'recov_smBeta':recovParams.smBeta,
                                        'gen_mu_0':genParams['mu_0'],
                                        'recov_mu_0':recovParams.mu_0,
                                        'gen_mu_1':genParams['mu_1'],
                                        'recov_mu_1':recovParams.mu_1,
                                        'gen_alpha':genParams['alpha'],
                                        'recov_alpha':recovParams.alpha},
                                       ignore_index=True)

        # Save the gen/recov parameters for this 'experiment'
        sampleDF.to_csv(f'{initDict.outDir}/sessions_{numSessions}_paramRecov.csv',index=False)

        # Compute the correlations between the generated and recovered parameters
        recovCorrDF = recovCorrDF.append({'sample_n': numTrials,
                                          'delta': np.corrcoef(sampleDF.gen_delta, sampleDF.recov_delta)[1,0],
                                          'smBeta': np.corrcoef(sampleDF.gen_smBeta, sampleDF.recov_smBeta)[1,0],
                                          'mu_0': np.corrcoef(sampleDF.gen_mu_0, sampleDF.recov_mu_0)[1,0],
                                          'mu_1': np.corrcoef(sampleDF.gen_mu_1, sampleDF.recov_mu_1)[1,0],
                                          'alpha': np.corrcoef(sampleDF.gen_alpha, sampleDF.recov_alpha)[1,0]},
                                         ignore_index=True)
    # Save parameter recover results
    recovCorrDF.to_csv(f'{initDict.homeDir}/paramRecov_results.csv', index=False)
    return


def unpackTask(genTask):
    # Get start of run
    runReset = []
    for i in np.arange(genTask.numSessions):
        runStart = np.zeros(len(genTask.sessionInfo[i].highChosen))
        runStart[0] = 1
        runReset = np.append(runReset, runStart)
    # Get choice attributes
    highChosen = np.hstack([genTask.sessionInfo[i].highChosen for i in np.arange(genTask.numSessions)])
    selectedStim = np.hstack([genTask.sessionInfo[i].selectedStim for i in np.arange(genTask.numSessions)])
    respIdx = np.hstack([genTask.sessionInfo[i].sessionResponses for i in np.arange(genTask.numSessions)])
    # Get reversal attributes
    reverseStatus = np.hstack([genTask.sessionInfo[i].reverseStatus for i in np.arange(genTask.numSessions)])
    reverseTrial = np.hstack([genTask.sessionInfo[i].reverseTrial for i in np.arange(genTask.numSessions)])
    # Get stimulus attributes
    pWin = np.hstack([genTask.sessionInfo[i].stimAttrib.pWin for i in np.arange(genTask.numSessions)])
    isHigh = np.hstack([genTask.sessionInfo[i].stimAttrib.isHigh for i in np.arange(genTask.numSessions)])
    isSelected = np.hstack([genTask.sessionInfo[i].stimAttrib.isSelected for i in np.arange(genTask.numSessions)])
    isWin = np.hstack([genTask.sessionInfo[i].stimAttrib.isWin for i in np.arange(genTask.numSessions)])

    # Get outcome attributes
    payOut = np.hstack([genTask.sessionInfo[i].payOut for i in np.arange(genTask.numSessions)])
    return dict2class(dict(runReset = runReset,
                           highChosen = highChosen,
                           selectedStim = selectedStim,
                           respIdx = respIdx,
                           reverseStatus = reverseStatus,
                           reverseTrial = reverseTrial,
                           pWin = pWin,
                           isHigh = isHigh,
                           isSelected = isSelected,
                           isWin = isWin,
                           payOut = payOut))

if __name__ == "__main__":
    runRecover()
