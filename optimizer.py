#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 13:38:45 2019

@author: vman
"""
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from joblib import Parallel, delayed
import multiprocessing
import time
from utilities import *
from Models.Model_params import fitParamContain

def unwrap_self(modFit, sIdx, seeds):
    return modFit.minimizer(sIdx, seeds)

class Optimizer(object):
    def __init__(self):
        self.tol = 1e-3
        self.maxiter = 1000
        self.disp = False
        return
    
    
    def minimizer(self, sIdx, seeds):
        optimResults = minimize(self.loss_fn,
                                seeds[:,sIdx],
                                method = 'BFGS',
                                tol = self.tol,
                                options = dict(disp = self.disp,
                                             maxiter = self.maxiter))
        return(optimResults)
    
    def computeBIC(self, NLL): 
        likelihood = np.exp(-1 * NLL)
        return self.mod.numParams * np.log(self.mod.numLLTrials) - 2 * np.log(likelihood)
        
    def computeAIC(self, NLL): 
        likelihood = np.exp(-1 * NLL)        
        return 2 * self.mod.numParams - 2 * np.log(likelihood)
    
    def runSerialOptimizer(self, seeds): 
        # Iterative across seed iterations
        serialResults = [] 
        for sIdx in np.arange(self.mod.numSeeds):
            print(f'Iteration: {sIdx} / {self.mod.numSeeds}')
            serialResults.append(self.minimizer(sIdx, seeds))                        
        return serialResults

    def runParallelOptimizer(self, seeds): 
        # Parallelize across seed iterations
        num_cores = multiprocessing.cpu_count()
        parallelResults = Parallel(n_jobs=num_cores)(delayed(unwrap_self)(self, sIdx, seeds)
                        for sIdx in np.arange(self.mod.numSeeds))
        return parallelResults
    
    def unpackOptimizer(self, parallelResults): 
        # Remove estimates that didn't converge z
        fitConv = np.array([parallelResults[s]['success'] for s in np.arange(len(parallelResults))])
        noConv_idx = np.where(fitConv == False)[0]
        print(f'{len(noConv_idx)} out of {self.mod.numSeeds} did not converge')
        parallelResults = np.delete(parallelResults, noConv_idx)
        
        # Store estimates from this seed iteration        
        fitParams = np.array([parallelResults[s]['x'] for s in np.arange(len(parallelResults))])
        fitNLL = np.array([parallelResults[s]['fun'] for s in np.arange(len(parallelResults))])
        fitInvHess = np.array([parallelResults[s]['hess_inv'] for s in np.arange(len(parallelResults))])            
        # Retrieve best fit parameters and assoc. NLL and InvHess
        minIdx = np.argmin(fitNLL)        
        optimParams = fitParams[minIdx]
        estimNLL = fitNLL[minIdx]
        estimAIC = self.computeAIC(estimNLL)
        estimBIC = self.computeBIC(estimNLL)        
        estimInvHess = fitInvHess[minIdx]
        # Transform best fit parameter
        estimParams = self.mod.transformParams(optimParams)            
        return fitParamContain(estimNLL, self.mod.numParams, estimAIC, estimBIC, estimInvHess), estimParams
    
        
    def tuneOptim(self, param):
        # Define likelihood function
        NLL = self.mod.likelihood(param)
        # Define tuning function
        normal_logpdf = lambda x: np.sum(np.log(norm.pdf(x, loc=0, scale=2)))        
        # Add log likelihood and log priors across parameters
        for p in np.arange(self.mod.numParams):
            NLL -= normal_logpdf(param[p])            
        return NLL
        
    def initSeeds(self):
        # Set up seed searching
        seeds = np.zeros((self.mod.numParams, self.mod.numSeeds), dtype='float')
        for i in np.arange(self.mod.numParams):
            seeds[i,:] = np.linspace(-5,5, self.mod.numSeeds)            
        self.loss_fn = self.tuneOptim
        return seeds
                 
    def fitModel(self, model, taskData):        
        # Initialize the model 
        self.mod = model()
        # Specify data for the model
        self.mod.inputData(taskData)
        # Initialize seeds    
        seeds = self.initSeeds()        
        ### Optimization     
        searchResults = self.runParallelOptimizer(seeds)        
        # Store estimates 
        optimRes, estimParams = self.unpackOptimizer(searchResults)                
        return optimRes, estimParams
    