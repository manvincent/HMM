#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 09:06:58 2022

@author: vman
"""

import pandas as pd
import numpy as np
np.seterr(over='ignore')
from scipy.stats import gamma, beta
from utilities import *
# Define model types
class ModelHMM(object):

    def __init__(self, emission_type = 'discrete'):
        self.numParams = 3
        self.numSeeds = 10**self.numParams        
        self.numStates = 2
        self.emission_type = emission_type
        self.param_bounds = dict2class(dict(delta=(0.01,0.99),
                                            beta=(1,20), # min, max
                                            pOut=(0.01,0.99) # parameter of p(out | state = incorrect)
                                            ))
        return

    def compute_prior(self, posterior, delta):
        '''
        Core hidden markov model function which updates the model on one trial.
        Compute current trial's prior state belief given transition probability
        from last trial

        Parameters
        ----------
        posterior : float
            Posterior probability of stim 0 being high on previous trial p(X_{t-1}) =  stim0_high
        delta : float
            Transition probability free parameter.

        Returns
        -------
        prior: float
            Prior probability of stim 0 being high on current trial p(X_{t}) =  stim0_high

        '''
        # Transition matrix is symmetrical so can be reduced to a vector for stay[0] vs switch[1] states
        A = np.array([1-delta, delta])
        # Update t-1 posterior to t prior via transition probability
        prior = A[0] * posterior + A[1] * (1-posterior)
        return prior

    def actor(self, prior, beta, alpha=0.5):
        """
        Args are the inputs to the model, besides the general model params:
        Args:
            prior: the current probability of being in state  p(X_{t}) =  stim0_high,
            computed by Bayesian update (def compute prior)
            beta: softmax inverse temperature; scalar
            alpha: fixed parameters for normalizing input to softmax
        """
        # Action selection through logistic function
        pOpt_1 =  1 / float( 1 + np.exp(beta * (prior - alpha)))
        ## Pick an action given the softmax p_switch: 0 = stim0, 1 = stim1
        pOptions = [1-pOpt_1, pOpt_1]
        ## Pick an option given the softmax probabilities
        respIdx = np.where(np.cumsum(pOptions) >= np.random.rand(1))[0][0]
        # output: 0 means left choice, 1 means right choice
        return respIdx, pOptions

    def compute_posterior(self, prior, p_0, p_1, obs, pOptions, action):
        '''
        Core hidden markov model function which updates the model on one trial.
        Compute current trial's prior state belief and updates according to
        action (stay/switch) and observed outcome.

        Parameters
        ----------
        prior : float
            Prior probability of stim 0 being high on current trial p(X_{t}) =  stim0_high
        p_0, p_1 : float
            Free parameters of the estimated p(Reward) given the correct action in each state
        obs : float
            Outcome of current trial
        pOptions : tuple
            Tuple indicating the probabilties of action=0 and action=1 from the softmax function
        action : int
            Scalar indicating the chosen action on t (response index)

        Returns
        -------
        posterior: float
            Posterior probability of stim 0 being high on current trial p(X_{t}) =  stim0_high

        '''

        # Compute t posterior given p(out | state)
        # Binarize outcome and action to use as indices
        obs_bin = (obs > 0) # Recode win/loss x 1/0 int for indexing

        # Discrete Emission probabilities
        # Compute p(action | state) for each state
        pAct_0 =  pOptions[action]
        pAct_1 = np.flip(pOptions)[action] # True because p(x=j) = (1 - p(x=i))
        # Compute p(reward | action, state) for each state, action
        if obs_bin: # if win
            if action == 0: 
                pRew_x0 = p_0
                pRew_x1 = (1 - p_1)
            elif action == 1: 
                pRew_x0 = (1 - p_0)
                pRew_x1 = p_1
        elif not obs_bin: # if loss
            if action == 0: 
                pRew_x0 = (1 - p_0)
                pRew_x1 = p_1
            elif action == 1: 
                pRew_x0 = p_0
                pRew_x1 = (1 - p_1)
        # Compute emission probabilities for each state
        pEmission_0 = pAct_0 * pRew_x0
        pEmission_1 = pAct_1 * pRew_x1
        posterior = prior * pEmission_0 / (pEmission_0 * prior + pEmission_1 * (1-prior))
        
        # Compute Q values 
        chosenQ =  (pEmission_0 * prior + pEmission_1 * (1-prior))
        
        # Unchosen p(a|x) and p(r|a,x) as 1 minus quantities for chosen
        pUnc_0 = pOptions[1-action]
        pUnc_1 = np.flip(pOptions)[1-action] 
        unchosenQ = ((1-pRew_x0) * pUnc_0 * prior + (1-pRew_x1) * pUnc_1 * (1-prior))
        Q = np.array([chosenQ, unchosenQ])                
        return posterior, Q

    def inputData(self, taskData):
        self.runReset = taskData.runReset
        self.numTrials = taskData.numTrials
        self.numLLTrials = taskData.numLLTrials        
        self.respIdx = taskData.respIdx
        self.reward = taskData.reward
        return

    def likelihood(self, param):
        # Transform optimizer parameters
        delta, beta, p_0 = self.transformParams(param)
        # Initialize posterior
        posterior = float(1) / self.numStates
        # Initilize trial likelihood
        LL = 0
        # Iterate over trials
        for tI in np.arange(self.numTrials):
            # If the start of a run, reset the q-values
            if self.runReset[tI] == 1:
                posterior = float(1) / self.numStates
            # If a response registered on this trial, update posterior
            if ~np.isnan(self.respIdx[tI]):
               # Compute new prior on this trial from last posterior
                prior = self.compute_prior(posterior, delta)
                # Compute action probabilities from the actor
                _, pOptions = self.actor(prior, beta)
                # Get observed response
                respIdx = self.respIdx[tI].astype(int)
                # Compute likelihood
                LL += np.log(pOptions[respIdx])
                # Observe outcome
                reward = 1 if self.reward[tI] > 0 else 0
                # Update posterior belief that selected action is correct
                if (~np.isnan(reward)):
                    posterior, _ = self.compute_posterior(prior,
                                                       p_0,
                                                       p_1,
                                                       reward,
                                                       pOptions,
                                                       respIdx)
            else:
                LL += 1e-10
        return -1*LL

    def transformParams(self, params):
        transParams = params
        # Transform offered values into model parameters
        transParams[0] = max(self.param_bounds.delta) / (1 + np.exp(-1 * transParams[0])) # delta
        transParams[1] = max(self.param_bounds.beta) / (1 + np.exp(-1 * transParams[1])) # smB
        transParams[2] = max(self.param_bounds.pOut) / (1 + np.exp(-1 * transParams[2])) # p_0
        return transParams
     
    def simulate(self, param): # For the 3-parameter model
        # Retrieve fitted param
        delta, beta, p_0 = param
        p_1 = p_0
        # Simulate using retrieved parameter to compute computational variables
        chosenPrior = np.ones(self.numTrials, dtype = float) * np.nan 
        unchosenPrior = np.ones(self.numTrials, dtype = float) * np.nan         
        chosenQ = np.ones(self.numTrials, dtype = float) * np.nan 
        unchosenQ = np.ones(self.numTrials, dtype = float) * np.nan 
        bayesPE = np.ones(self.numTrials, dtype = float) * np.nan   
        simResp = np.ones(self.numTrials, dtype = float) * np.nan 
        
        # Initialize posterior
        posterior = float(1) / self.numStates        
        
        for tI in np.arange(self.numTrials):
            if self.runReset[tI] == 1:
                # Initialize posterior probabilities
                posterior = float(1) / self.numStates
            
            if ~np.isnan(self.respIdx[tI]):            
                # Compute new prior on this trial from last posterior
                prior = self.compute_prior(posterior, delta)
                # Compute action probabilities from the actor
                simResp[tI], pOptions = self.actor(prior, beta)
                                                                
                # Retreive observed (empirical) choice                
                respIdx = self.respIdx[tI].astype(int)
                # Observe outcome
                reward = 1 if self.reward[tI] > 0 else 0
                
                # Get prior of chosen option
                if respIdx == 0:
                    chosenPrior[tI] = prior
                    unchosenPrior[tI] = 1 - prior
                elif respIdx == 1: 
                    chosenPrior[tI] = 1 - prior
                    unchosenPrior[tI] = prior
                                                
                if ~np.isnan(reward):
                    # Update posterior belief that selected action is correct
                    posterior, (chosenQ[tI], unchosenQ[tI]) = self.compute_posterior(prior,
                                                                                     p_0,
                                                                                     p_1,
                                                                                     reward, 
                                                                                     pOptions,
                                                                                     respIdx)
                    # Compute bayesian suprise
                    bayesPE[tI] = posterior - prior 
                    
                                        
        # Store simulated values as dataframe
        simDF = pd.DataFrame(dict(chosenPrior=chosenPrior,
                                  unchosenPrior=unchosenPrior,                                  
                                  chosenQ=chosenQ,
                                  unchosenQ=unchosenQ,
                                  bayesPE=bayesPE,
                                  simResp=simResp)).round(4)                    
        return simDF
                


        