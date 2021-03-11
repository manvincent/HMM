#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 13:38:45 2019

@author: vman
"""
import numpy as np
from scipy.optimize import minimize
from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()-1
import time
from defineModel import *
from utilities import *

def unwrap_self(modFit, seedIter, seeds, taskData, numTrials, priors):
    return modFit.minimizer(seedIter, seeds, taskData, numTrials, priors)


class Optimizer(object):
    def __init__(self):
        self.xtol = 0.001
        self.ftol = 0.01
        self.maxiter = 1000
        self.disp = False
        return

    def getFit(self, taskData, numTrials, payOut, emission_type):
        # Initialize the model and bounds
        self.initMod = ModelType(payOut, emission_type)
        # Define parameter bounds
        self.boundList = tuple((self.initMod.delta_bounds,
                                self.initMod.smBeta_bounds,
                                self.initMod.mu_0_bounds,
                                self.initMod.mu_1_bounds,
                                self.initMod.sigma_bounds))

        # Define prior distribution
        priors = self.initMod.paramPriors()
        ### Optimization
        # Set up seed searching
        numSeeds = 500
        seeds = np.zeros([self.initMod.numParams,numSeeds], dtype=float)
        seeds[0,:] = np.random.permutation(np.linspace(self.initMod.delta_bounds[0], self.initMod.delta_bounds[1], numSeeds))
        seeds[1,:] = np.random.permutation(np.linspace(self.initMod.smBeta_bounds[0], self.initMod.smBeta_bounds[1], numSeeds))
        seeds[2,:] = np.random.permutation(np.linspace(self.initMod.mu_0_bounds[0], self.initMod.mu_0_bounds[1], numSeeds))
        seeds[3,:] = np.random.permutation(np.linspace(self.initMod.mu_1_bounds[0], self.initMod.mu_1_bounds[1], numSeeds))
        seeds[4,:] = np.random.permutation(np.linspace(self.initMod.sigma_bounds[0], self.initMod.sigma_bounds[1], numSeeds))
        # Parallelize across seed iterations
        num_cores = multiprocessing.cpu_count()
        # parallelResults = Parallel(n_jobs=num_cores)(delayed(unwrap_self)(self, seedIter, seeds, taskData, numTrials, priors)
        #                 for seedIter in np.arange(numSeeds))
        # Non-parallel version for debugging
        a = []
        for seedIter in np.arange(numSeeds):
            print(f'Fitting seed {seedIter}')
            a.append(self.minimizer(seedIter, seeds, taskData, numTrials, priors))
        return parallelResults

    def minimizer(self, seedIter, seeds, taskData, numTrials, priors):
        optimResults = minimize(self.posterior,
                                seeds[:,seedIter],
                                args = (taskData, numTrials, priors),
                                method = 'TNC',
                                bounds = self.boundList,
                                options = dict(disp = self.disp,
                                             maxiter = self.maxiter,
                                             xtol = self.xtol,
                                             ftol = self.ftol))
        return(optimResults)

    def likelihood(self, param, taskData, numTrials):
        # Unpack parameter values
        delta, smBeta, mu_0, mu_1, sigma = param
        # Initialize posterior
        posterior = float(1) / self.initMod.numStates
        # Get first (random) observed action
        respIdx = taskData.respIdx[0]
        # Initilize trial likelihood
        pChoice = 0
        # Iterate over trials
        for tI in np.arange(1, numTrials):
            if taskData.runReset[tI] == 1:
                # Initialize posterior probabilities
                posterior = float(1) / self.initMod.numStates

            # Run model simulation
            _, pOptions = self.initMod.actor(posterior, smBeta, respIdx)

            # Get observed response
            respIdx = taskData.respIdx[tI].astype(int)

            if ~np.isnan(taskData.respIdx[tI]):
                # Get likelihood of data | model
                pChoice += np.log(pOptions[respIdx])
                # Create tuple of observed responses actions (t-1, t)
                actions = np.array([taskData.respIdx[tI-1],
                                     respIdx])

                # Observe outcome
                reward = taskData.payOut[tI]
                # Update posterior belief that selected action is correct
                if (~np.isnan(reward)):
                    posterior = self.initMod.belief_propogation(posterior,
                                                                delta,
                                                                mu_0,
                                                                mu_1,
                                                                sigma,
                                                                reward, actions)
            else:
                pChoice += 1e-10

        return pChoice

    def posterior(self, param, taskData, numTrials, priors):
        # Define likelihood function
        logLike = self.likelihood(param, taskData, numTrials)
        # Add log likelihood and log priors across parameters
        # logLike += priors['delta_prior'](param[0])
        # logLike += priors['smBeta_prior'](param[1])
        # logLike += priors['sigma_prior'](param[4])
        # Compute posterior (neg log) likelihood
        NLL = -1 * logLike
        return NLL
