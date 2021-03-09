#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 18:47:26 2021

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
os.chdir('path/to/project')

from optimizer import *
from utilities import *

### Recovery script

def runRecover():
    numMaxDays = 1
    sessPerDays = 5
    fitModel(numMaxDays, sessPerDays)


def initRecover():
    modelName = 'cond_action'
    emission_type = 'discrete'
    homeDir = 'path/to/project'
    outDir = f'{homeDir}/Fit_results_{modelName}_{emission_type}_MLE'
    if not os.path.exists(outDir):
        os.makedirs(outDir)
    # Initialize the Model class
    initDict = dict2class(dict(modelName = modelName,
                               emission_type = emission_type,
                               homeDir = homeDir,
                               outDir = outDir))
    return(initDict)


def fitModel(numMaxDays, sessPerDays):

    # Specify experiment info
    #subList = np.array([2,3,4,5])
    subList = np.array([3])
    #numDays = np.array([6,4,3,1])
    numDays = 1
    # Initialize the model fitting object
    initDict = initRecover()
    # Create structures for storing estimates across subjects
    sampleDF = pd.DataFrame()
    for subIdx, subID in enumerate(subList):
        # Load in data
        inFile = f'sub{subID}_data.csv'
        subDF = pd.read_csv(f'{initDict.homeDir}/{inFile}')
        for day in np.arange(numDays)+1:
            print(f'Fitting parameters for subID: {subID}, day: {day}')
            dayDF = subDF[subDF.dayID == day]
            # Unpack the task sessions
            taskData = unpackTask(dayDF)
            numTrials = taskData.numTrials
            initOptimizer = Optimizer()
            # Run the optimizer
            parallelResults = initOptimizer.getFit(taskData, numTrials, initDict.emission_type)
            # Store estimates from this seed iteration
            fitParams = np.array([parallelResults[s]['x'] for s in np.arange(len(parallelResults))])
            fitNLL = np.array([parallelResults[s]['fun'] for s in np.arange(len(parallelResults))])
            # Store best fitted parameter likelihoods and values
            minIdx = np.argmin(fitNLL)
            fitParams = fitParamContain(fitNLL[minIdx]).action_HMM_fitParams(fitParams[minIdx])

            # Append fitted parameters to dataframe
            sampleDF = sampleDF.append({'subID': int(subID),
                                        'day': int(day),
                                        'fit_delta':fitParams.delta,
                                        'fit_smBeta':fitParams.smBeta,
                                        'fit_mu_0':fitParams.mu_0,
                                        'fit_mu_1':fitParams.mu_1,
                                        'fit_alpha':fitParams.alpha}, ignore_index=True)


            # Simulate using retrieved parameter to compute computational variables
            initMod = ModelType(initDict.emission_type)
            posterior_correct = np.ones(numTrials, dtype = float) * np.nan
            simResp = np.ones(numTrials, dtype = int) * np.nan

            # Initialize posterior
            posterior = posterior_correct[0] = float(1) / initMod.numStates
            # Get first (random) observed action
            respIdx = simResp[0] = taskData.respIdx[0]

            for tI in np.arange(1,numTrials):
                if taskData.runReset[tI] == 1:
                    # Initialize posterior probabilities
                    posterior = float(1) / initMod.numStates

                # Have model simulate response given parameters
                simResp[tI], _ = initMod.actor(posterior, fitParams.smBeta, fitParams.alpha, respIdx)

                # Retreive observed (empirical) choice
                respIdx = taskData.respIdx[tI]

                if ~np.isnan(respIdx):
                    # Create tuple of observed responses actions (t-1, t)
                    actions = np.array([taskData.respIdx[tI-1],
                                        respIdx])
                    # Observe outcome
                    reward = 1 if taskData.payOut[tI] > 0 else 0
                    # Update posterior belief that selected action is correct
                    if (~np.isnan(reward)):
                        posterior_correct[tI] = posterior = initMod.belief_propogation(posterior,
                                                                    fitParams.delta,
                                                                    fitParams.mu_0,
                                                                    fitParams.mu_1,
                                                                    reward, actions)

            # Create dataframe of model-based variables
            compVarDF = pd.DataFrame({'fit_delta':fitParams.delta,
                                      'fit_smBeta':fitParams.smBeta,
                                      'fit_mu_0':fitParams.mu_0,
                                      'fit_mu_1':fitParams.mu_1,
                                      'fit_alpha':fitParams.alpha,
                                      'posterior': posterior_correct,
                                      'simResp': simResp})
            compVarDF.to_csv(f'{initDict.outDir}/sub-{subID}_day-{day}_modelvars.csv', index = False)
    # Save the gen/recov parameters for this 'experiment'
    sampleDF.to_csv(f'{initDict.outDir}/group_modelparams.csv',index=False)
    return





def unpackTask(taskDF):
    # Get choice attributes
    highChosen = np.array(taskDF.highChosen, dtype=bool)
    selectedStim = np.array(taskDF.response_stimID)
    respIdx = selectedStim - 1

    # Get reversal attributes
    reverseStatus = np.array(taskDF.reverseStatus, dtype=bool)
    reverseTrial = np.array(taskDF.reverseTrial, dtype=bool)

    # Get stimulus attributes
    pWin = np.array([taskDF.stim1_pWin, taskDF.stim2_pWin])
    isHigh = np.array([taskDF.stim1_High, taskDF.stim2_High], dtype=bool)
    isSelected = np.array([taskDF.selected_stim1, taskDF.selected_stim2])
    isWin = np.array([taskDF.stim1_isWin, taskDF.stim2_isWin])

    # Get outcome attributes
    outMag = taskDF.outMag / 100

    # Get session and trial lists
    sessID = np.array(taskDF.sessNo)
    trialNo = np.array(taskDF.trialNo)
    numTrials = len(taskDF)
    absTrials = np.arange(numTrials)

    # Get start of run
    runReset = np.zeros(numTrials)
    runReset[np.where(trialNo == 1)] = 1


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
                           payOut = outMag,
                           sessID = sessID,
                           trialNo = trialNo,
                           numTrials = numTrials,
                           absTrials = absTrials
                           ))

if __name__ == "__main__":
    runRecover()
