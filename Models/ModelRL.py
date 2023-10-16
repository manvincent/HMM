import pandas as pd
import numpy as np
from scipy.stats import gamma, beta, uniform
from utilities import *
# Define model types
class ModelRL(object):

    def __init__(self):
        self.numParams = 2
        self.numSeeds = 10**self.numParams        
        self.param_bounds = dict2class(dict(alpha=(0.01,0.99),
                                            beta=(1,20) # min, max
                                            ))
        return

    def learner(self, Q, alpha, reward):
        """
        Args are the inputs to the model, besides the general model params:
        Args:
            Q: Model learnt Q value of chosen alternative
            alpha: delta-rule learning rate; scalar
            reward: observed reward outcome 
        """
        RPE = (reward - Q)
        Q =  Q + alpha * RPE
        return(Q, RPE)
    
    def actor(self, Q, beta):
        """
        Args are the inputs to the model, besides the general model params:
        Args:
            Q: the expected action value, computed by learner (for all choices; vector)
            beta: softmax inverse temperature; scalar
        """
        # Specify number of parameters
        reward = int()
        RPE = int()
        # Action selection through logistic function
        pOpt1 = 1 / float( 1 + np.exp(beta *(Q[1] - Q[0])))
        pOptions = np.array([pOpt1,1-pOpt1])
        '''
        ## Note: if left response is assigned to index 0 and right to index 1 
        this returns p(right choice)
        '''
        ## Pick an option given the softmax probabilities
        respIdx = np.where(np.cumsum(pOptions) >= np.random.rand(1))[0][0]
        # output: 0 means left choice, 1 means right choice
        return(respIdx, pOptions)
    
    def inputData(self, taskData):
        self.runReset = taskData.runReset
        self.numTrials = taskData.numTrials 
        self.numLLTrials = taskData.numLLTrials        
        self.respIdx = taskData.respIdx
        self.reward = taskData.reward
        return
        
    def likelihood(self, param):
        # Transform optimizer parameters
        # qA, smB = param
        qA, smB = self.transformParams(param)
        # Initialize Q values
        qval = np.ones(2, dtype = float) * 0.5
        # Initilize trial likelihood
        LL = 0
        for tI in np.arange(self.numTrials):
            # If the start of a run, reset the q-values 
            if self.runReset[tI] == 1: 
                qval = np.ones(2, dtype = float) * 0.5
            # If a response registered on this trial, compute likelihood    
            if ~np.isnan(self.respIdx[tI]):
                # Run model simulation
                _, pOptions = self.actor(qval, smB)
                # Get observed response
                respIdx = self.respIdx[tI].astype(int)
                # Get likelihood of data | model
                LL = np.nansum([LL, np.log(pOptions[respIdx])])
                # Observe outcome
                reward = self.reward[tI]
                # Update action value according to the delta rule
                qval[respIdx], _ = self.learner(qval[respIdx], qA, reward)                                
            else:
                LL += 1e-10
        return -1*LL
    
    def transformParams(self, params):
        transParams = params
        # Transform offered values into model parameters
        transParams[0] = max(self.param_bounds.alpha) / (1 + np.exp(-1 * transParams[0])) # qA
        transParams[1] = max(self.param_bounds.beta) / (1 + np.exp(-1 * transParams[1])) # smB        
        return transParams
    
    def simulate(self, param):
        # Retrieve fitted param
        qA, smB = param
        # Simulate using retrieved parameter to compute computational variables
        chosenQ = np.ones(self.numTrials, dtype = float) * np.nan 
        unchosenQ = np.ones(self.numTrials, dtype = float) * np.nan 
        simResp = np.ones(self.numTrials, dtype = float) * np.nan 
        RPE = np.ones(self.numTrials, dtype = float) * np.nan         
        # Initialize qval
        qval = np.ones(2, dtype = float) * 0.5
        for tI in np.arange(self.numTrials):
            # If the start of a run, reset the q-values 
            if self.runReset[tI] == 1: 
                qval = np.ones(2, dtype = float) * 0.5
            # Retreive observed (empirical) choice
            respIdx = self.respIdx[tI].astype(int)
            unchosenIdx = (1-self.respIdx[tI]).astype(int)
            # Have model simulate response given parameters
            if ~np.isnan(self.respIdx[tI]):
                # Run model simulation
                simResp[tI], _ = self.actor(qval, smB)            
                # Store computational variables
                chosenQ[tI] = qval[respIdx]
                unchosenQ[tI] = qval[unchosenIdx]
                # Retrieve observed reward
                reward = self.reward[tI]
                # Update learner 
                qval[respIdx], RPE[tI] = self.learner(qval[respIdx], qA, reward)
        # Store simulated values as dataframe
        simDF = pd.DataFrame(dict(chosenQ=chosenQ,
                                  unchosenQ=unchosenQ,
                                  RPE=RPE,
                                  simResp=simResp)).round(4)
        return simDF
