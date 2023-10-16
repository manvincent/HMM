#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 11:07:54 2022

@author: vman
"""

# Set up a container for model parameter estimates (fits)
class fitParamContain():
    def __init__(self,  NLL, numParams, AIC, BIC, invHess):
        self.nll = NLL 
        self.numParams = numParams
        self.AIC = AIC
        self.BIC = BIC
        self.invHess = invHess        
        return
    
    def HMM_dj(self, params):
        self.delta= params[0]
        self.beta = params[1]
        self.p_0 = params[2]
        self.p_1 = params[3]
        return
    
    def RL_rw(self, params):
        self.alpha = params[0]
        self.beta = params[1]
        return
    
    def RL_forget(self, params):
        self.alpha = params[0]
        self.beta = params[1]
        self.eta = params[2]        
        return 
    
    def RL_fictive(self, params):
        self.alpha = params[0]
        self.beta = params[1]
        return 