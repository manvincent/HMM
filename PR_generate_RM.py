#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 22:57:33 2021

@author: vman
"""


## Make sure to activate the Intel Python distribution
## (bash): source activate IDP

import numpy as np
import os
os.chdir('/home/vman/Dropbox/PostDoctoral/gitRepos/HMM')
import itertools
from defineModel import *
from utilities import *


# Generate data, loop across numSession lengths:
def runGenerate():
    numMaxDays = 6
    sessPerDays = 5
    for numSessions in np.arange(1,numMaxDays+1) * sessPerDays: 
        print(f'Simulation with {numSessions} sessions')
        genData(numSessions)
        
### Generate script
def initGenerate(numSessions):
    ###### Global task properties ######
    # Defining directories #
    homeDir = '/home/vman/Dropbox/PostDoctoral/gitRepos/HMM'
    if not os.path.exists(homeDir): 
        os.mkdir(homeDir)
    outDir = f'{homeDir}/Generate/Sessions_{numSessions}'
    if not os.path.exists(outDir):
        os.makedirs(outDir)
    # Initialize the Model class
    outMag = np.array([0.25, 0.50, 0.75])
    payOut = np.array([outMag, -1*outMag])
    initMod = ModelType(payOut, emission_type='gaussian')
    # Initialize the dictionary 
    initDict = dict2class(dict(outDir = outDir))
    return(initDict,initMod)

def genData(numSessions):
    # Number of different parameter values
    numParamEst = 10
    # Intialize structures and set up directories
    [initDict,initMod] = initGenerate(numSessions)
    # Number of simulations
    numIter = 1000
    # Set up range of model parameter values
    modelStruct = dict()
    # Generative parameters ()
    genVal  = initMod.genPriors(numParamEst)
    param_names = list(genVal.keys())
    
    sub_params = np.empty((numIter, initMod.numParams), dtype=float)
    for subID in np.arange(numIter): 
        sub_params[subID,:] = [np.random.choice(list(genVal.values())[i]) for i in np.arange(initMod.numParams)]
        
    for subID in np.arange(numIter):
        # Specify model parameter for the current sim
        genParams = {} 
        for param_idx in np.arange(initMod.numParams):
            genParams.update({param_names[param_idx]: sub_params[subID,param_idx]})
            
        # Set up the task parameters
        initDict = initTask(subID+1, initDict, numSessions)

        # Loop through sessions and t rials
        for sI in np.arange(initDict.numSessions):
            # Initialize session information
            sessionInfo = initDict.sessionInfo[sI]
            # Initialize the reverseStatus to False (need participant to get 4 continuous correct)
            reverseStatus = False
            # Initialize posterior probabilities 
            posterior = initDict.sessionInfo[sI].posterior_correct[0] = float(1) / initMod.numStates            
            # Initialize first random action 
            respIdx = initDict.sessionInfo[sI].sessionResponses[0] = np.random.choice([0,1])                            
            # Iterate over trials
            for tI in np.arange(1, initDict.trialsPerSess):
                # Initialize trials
                initTrial(tI, initDict, sessionInfo)               
                # Make response given posterior on last trial
                respIdx, _ = initMod.actor(posterior, genParams['smBeta'], respIdx)                                
                initDict.sessionInfo[sI].sessionResponses[tI] = respIdx
                # Create tuple of actions (t-1, t)
                actions = np.array([initDict.sessionInfo[sI].sessionResponses[tI-1],
                                    respIdx])
                # Compute outcome 
                reward = computeOutcome(tI, initDict, sessionInfo, respIdx) 
                # Update posterior belief that selected action is correct
                if (~np.isnan(reward)):
                    posterior = initMod.belief_propogation(posterior,
                                                           genParams['delta'],
                                                           genParams['mu_0'],
                                                           genParams['mu_1'],
                                                           genParams['sigma'],
                                                           reward,
                                                           actions)
                    
                initDict.sessionInfo[sI].posterior_correct[tI] = posterior                
                # Compute reversal 
                reverseStatus = computeReversal(tI, initDict, sessionInfo, reverseStatus)
                
        # Store simulations
        modelStruct = dict2class(dict(genParams = genParams))
        # Convert data
        outPack = convertSave(initDict, modelStruct)
        # Save data (for current session)
        save_obj(outPack, initDict.outDir + os.sep + 'sim_' + str(initDict.subID))
        # Save as .mat files (for current session)
    return



def stimParam(initDict):
    # For the first axis, indices of 0 = stim1 and 1 = stim2
    pWin = np.empty((2,initDict.trialsPerSess), dtype=float)
    isHigh = np.empty((2,initDict.trialsPerSess), dtype=bool)
    isSelected = np.empty((2,initDict.trialsPerSess), dtype=float)
    isWin = np.empty((2,initDict.trialsPerSess), dtype=float)
    outMag = np.empty((2,initDict.trialsPerSess), dtype=float)
    return dict(pWin=pWin,
                isHigh=isHigh,
                isSelected=isSelected,
                isWin=isWin,
                outMag=outMag)
            
def initTask(subID, initDict, numSessions):
    # Specify task parameters
    pWinHigh = 0.65
    pWinLow = 0.35
    pReversal = 0.25
    outMag = np.array([0.25, 0.50, 0.75])    
    # Set up the session-wise design
    trialsPerSess = 60
    # Flatten as dict2class    
    initDict.__dict__.update({
            'pWinHigh': pWinHigh,
            'pWinLow': pWinLow,
            'outMag': outMag,
            'pReversal': pReversal, 
            'subID':subID,
            'numSessions':numSessions,
            'trialsPerSess':trialsPerSess})
    sessionInfo = np.empty(initDict.numSessions, dtype=object)
    for sI in np.arange(numSessions):
        # Initialize (before first reversal) which stim is p(high)
        stim1_high = np.random.binomial(1, 0.5, 1).astype(bool)[0]
        # Store whether the good (pWinHigh) option was chosen
        highChosen = np.zeros(initDict.trialsPerSess,dtype=bool)
        # Store which stim is the selected stim
        selectedStim = np.zeros(initDict.trialsPerSess,dtype=int)
        # Store whether reversals are possible on trial tI
        reverseStatus = np.zeros(initDict.trialsPerSess,dtype=bool)
        # Store whether a reversal occurred on trial tI
        reverseTrial = np.zeros(initDict.trialsPerSess,dtype=bool)
        # Create vector of reversal status after reversals possible
        reverseVec = np.zeros(int(np.floor(1/pReversal)), dtype=bool)
        reverseVec[0] = True        
        # Initialize timing containers
        sessionResponses = np.empty(initDict.trialsPerSess)        
        # Initialize stim attribute containers
        stimAttrib = dict2class(stimParam(initDict))
        # Initialize payout container
        payOut = np.zeros(initDict.trialsPerSess,dtype=float)
        # Initialize the posterior probability (that selected choice is correct)
        posterior_correct = np.empty(initDict.trialsPerSess,dtype=float)
        
        # Flatten into class object
        sessionInfo[sI] = dict2class(dict(stim1_high=stim1_high,
                                       highChosen=highChosen,
                                       selectedStim=selectedStim,
                                       reverseStatus=reverseStatus,
                                       reverseTrial=reverseTrial,
                                       reverseVec=reverseVec,
                                       sessionResponses=sessionResponses,
                                       stimAttrib=stimAttrib,
                                       payOut=payOut,
                                       posterior_correct=posterior_correct))
    initDict.__dict__.update({'sessionInfo':sessionInfo})
    return(initDict)



def computeOutcome(tI, initDict, sessionInfo, respIdx):
     # Draw win and loss magnitudes
     outMag = np.random.choice(initDict.outMag)
     # Determine which stim was chosen
     if (respIdx == 0):
         sessionInfo.selectedStim[tI] = 1
         pWin = sessionInfo.stimAttrib.pWin[respIdx,tI]
         isWin = np.random.binomial(1,pWin,1).astype(bool)[0]
     elif (respIdx == 1):
         sessionInfo.selectedStim[tI] = 2
         pWin = sessionInfo.stimAttrib.pWin[respIdx,tI]
         isWin = np.random.binomial(1,pWin,1).astype(bool)[0]          
     # Record stim attributes
     sessionInfo.stimAttrib.isSelected[respIdx, tI] = 1
     sessionInfo.stimAttrib.isWin[respIdx, tI] = isWin
     sessionInfo.stimAttrib.outMag[respIdx, tI] = outMag
     sessionInfo.stimAttrib.isSelected[1-respIdx, tI] = 0
     sessionInfo.stimAttrib.isWin[1-respIdx, tI] = np.nan
     sessionInfo.stimAttrib.outMag[1-respIdx, tI] = np.nan
     # Record whether they chose the high value option
     sessionInfo.highChosen[tI] = True if (pWin == initDict.pWinHigh) else False
     # Record the observed payOut
     sessionInfo.payOut[tI] = reward = outMag if isWin else -1*outMag
     return reward


def initTrial(tI, initDict, sessionInfo):
    # Compute win probabilities for each stim
    if (sessionInfo.stim1_high):
        # Toggle which stim is high/low
        sessionInfo.stimAttrib.pWin[0, tI] = initDict.pWinHigh
        sessionInfo.stimAttrib.isHigh[0, tI] = True
        sessionInfo.stimAttrib.pWin[1, tI] = initDict.pWinLow
        sessionInfo.stimAttrib.isHigh[1, tI] = False
    else:
        # Toggle which stim is high/low
        sessionInfo.stimAttrib.pWin[0, tI] = initDict.pWinLow
        sessionInfo.stimAttrib.isHigh[0, tI] = False
        sessionInfo.stimAttrib.pWin[1, tI] = initDict.pWinHigh
        sessionInfo.stimAttrib.isHigh[1, tI] = True
    return

def computeReversal(tI, initDict, sessionInfo, reverseStatus):
    # No reversals in the first 4 trials of the task
    if (tI < 3):
        sessionInfo.reverseTrial[tI] = False
    # After the first 4 trials, reversals are possible
    if (tI >= 3):
        # Reversals are possible if 4 continuous correct responses
        if (np.all(sessionInfo.highChosen[tI-3:tI+1] == True)) and (np.all(np.diff(sessionInfo.selectedStim[tI-3:tI+1]) == 0)):
            reverseStatus = True
        # If 4 continuous incorrect responses, not sufficient learning. Reset reversalStatus
        if (np.all(sessionInfo.highChosen[tI-3:tI+1] == False)):
            reversalStatus = False
        # If reversals are possible
        sessionInfo.reverseStatus[tI] = reverseStatus
        # Store the reversal status of the trial
        if (reverseStatus):
            # Determine whether reversals occurs on this trials
            reverse = np.random.binomial(1, initDict.pReversal, 1).astype(bool)[0]
            if (reverse):
                # Execute high stim reversal
                sessionInfo.stim1_high = not sessionInfo.stim1_high
                sessionInfo.reverseTrial[tI] = True
                # Reset the reverseStatus
                reverseStatus = False
        else: 
            sessionInfo.reverseTrial[tI] = False
            
    return reverseStatus


# Execute    
if __name__ == "__main__":
    runGenerate()
        
