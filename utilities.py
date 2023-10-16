from copy import deepcopy as dcopy
import numpy as np
import pickle

import contextlib
import functools
import time

# Additional necessary functions (from config.py)
class dict2class(object):
    """
    Converts dictionary into class object
    Dict key,value pairs become attributes
    """
    def __init__(self, dict):
        for key, val in dict.items():
            setattr(self, key, val)
            
            
def counterbalance(subID):
    if is_odd(subID):
        sub_cb = 1
    elif not is_odd(subID):
        sub_cb = 2
    return(sub_cb)


def is_odd(num):
   return num % 2 != 0


def minmax(array): 
    return (array - min(array)) / (max(array) -min(array))


def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name ):
    with open(name, 'rb') as f:
        return pickle.load(f)


def unpackTask(taskDF, initDict, remove_aborts = False):
    if initDict.species == 'human':
        selectedStim = np.array(taskDF.choice)
        isAbort = np.zeros(len(taskDF))
        reward = np.array(taskDF.reward)
    elif initDict.species == 'monkey':
        if remove_aborts:
            taskDF = taskDF[taskDF.isAbort == 0]
        selectedStim = np.array(taskDF.response_stimID)
        isAbort = np.array(taskDF.isAbort, dtype=bool)
        reward = np.array(taskDF.isWin)
    # Get choice attributes
    highChosen = np.array(taskDF.highChosen, dtype=bool)
    respIdx = selectedStim - 1
    # Get reversal attributes
    reverseTrial = np.array(taskDF.reverseTrial, dtype=bool)
    stimHigh = taskDF.stimHigh
    # Get stimulus attributes
    pWin = taskDF.Preward
    # Get session and trial lists
    sessNo = np.array(taskDF.sessNo)
    trialNo = np.array(taskDF.trialNo)
    # Get within-block trial number
    blockNo = taskDF.block
    blockTrialNo = taskDF.blockTrialNo
    # Get absolute trial count and number across data
    numTrials = len(taskDF)
    numLLTrials = numTrials - np.sum(isAbort)
    absTrialNo = np.arange(numTrials)+1
    # Get start of run
    runReset = np.zeros(numTrials)
    runReset[np.where(trialNo == 1)] = 1
    return dict2class(dict(highChosen=highChosen,
                           selectedStim=selectedStim,
                           respIdx=respIdx,
                           isAbort=isAbort,
                           reverseTrial=reverseTrial,
                           stimHigh=stimHigh,
                           pWin=pWin,
                           reward=reward,
                           sessNo=sessNo,
                           trialNo=trialNo,
                           blockNo=blockNo,
                           blockTrialNo=blockTrialNo,
                           numTrials=numTrials,
                           numLLTrials=numLLTrials,
                           absTrialNo=absTrialNo,
                           runReset=runReset))
