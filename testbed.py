#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: vman
"""
import numpy as np
import pandas as pd 
import scipy.stats
import os
os.chdir('/home/vman/Dropbox/PostDoctoral/gitRepos/HMM')

#data = pd.read_csv('sub2_data.csv')

# Make some synthetic data 
# Parameterse 
nTrial = 200 
block_length = 20 
pWin = 0.8
nTrial_persistent = 5

true_state = np.tile(np.concatenate([np.ones(block_length), 
                                     np.zeros(block_length)]), 
                     nTrial // block_length // 2)

  
action_one = np.ones(block_length)
action_one[:5] = 0 

action_zero = np.ones(block_length) * 0
action_zero[:5] =1
actions =  np.tile(np.concatenate([action_one,
                                   action_zero]),                                                        
                     nTrial // block_length // 2)

obs = np.empty(len(actions), dtype = int)
for t in np.arange(len(actions)): 
    if actions[t] == true_state[t]: 
        obs[t] = np.random.binomial(1,pWin)
    else: 
        obs[t] = np.random.binomial(1,1-pWin)
        

numStates = 2
delta = 0.25

A_switch = np.array([[delta, 1-delta], 
                     [1-delta, delta]])  
A_stay = 1 - A_switch

B_correct = np.array([-0.15, 0.35])
B_incorrect = -np.flip(B_correct)   

# Gaussian Emission probabilities 
            
pi = np.array([0.5, 0.5])

prior = 0

prior_output = [] 
post_output = [] 
posterior = 0.5

for t in np.arange(len(obs)):
    # Compute prior of trial t conditioned on action 
    if actions[t] == actions[t-1]: # If stay action:
        prior = A_stay[0,0] * posterior + A_stay[0,1] * (1-posterior)
    elif actions[t] != actions[t-1]: # If switch action:
        prior = A_switch[0,0] * posterior + A_switch[0,1] * (1-posterior)
    # Compute posterior of trial t
    # posterior = prior * B_correct[obs[t]] / (B_correct[obs[t]] * prior + B_incorrect[obs[t]] * (1-prior))
    B_incorrect = 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-1/2 * ((obs[t] - mu_0)/sigma)**2)        
    B_correct = 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-1/2 * ((obs[t] - mu_1)/sigma)**2)
    posterior = prior * B_correct / (B_correct * prior + B_incorrect * (1-prior))
    
    

        
    prior_output.append(prior)
    post_output.append(posterior)
    
    
output = np.array(post_output) # belief in 'correct' state 

inferred_state = np.ones(len(true_state))
inferred_state[np.where(output > 0.5)[0]] = actions[np.where(output > 0.5)[0]]
inferred_state[np.where(output < 0.5)[0]] = 1 - actions[np.where(output < 0.5)[0]]

%matplotlib qt5
plt.plot(true_state); 
plt.scatter(np.arange(200), actions, alpha=0.2); 
plt.plot(output, c='red', alpha=0.5);
plt.scatter(np.where(obs)[0], actions[np.where(obs)], alpha=0.6, color='purple');
plt.plot(inferred_state, alpha=.6)





''' Example where model generates actiosn '''

# Make some synthetic data 
# Parameterse 
nTrial = 200 
block_length = 20 
pWin = 0.8
nTrial_persistent = 5

true_state = np.tile(np.concatenate([np.ones(block_length), 
                                     np.zeros(block_length)]), 
                     nTrial // block_length // 2)

  
# action_one = np.ones(block_length)
# action_one[:5] = 0 

# action_zero = np.ones(block_length) * 0
# action_zero[:5] =1
# actions =  np.tile(np.concatenate([action_one,
#                                    action_zero]),                                                        
#                      nTrial // block_length // 2)

obs = np.empty(200, dtype=float)

numStates = 2
delta = 0.25

A_switch = np.array([[delta, 1-delta], 
                     [1-delta, delta]])  
A_stay = 1 - A_switch

sigma = 3
mu_0 = -0.15
mu_1 = 0.35
beta = 20
B_correct = np.array([mu_0, mu_1])
B_incorrect = -np.flip(B_correct)   

# Gaussian Emission probabilities 
            
pi = np.array([0.5, 0.5])

prior = 0

posterior = 0.5
action_vector = [] 
actions = np.ones(len(obs), dtype = int)
obs = np.empty(len(actions), dtype = int) * 0
infer = np.empty(len(actions), dtype = int) 

prior_output = np.ones(len(actions), dtype = int) * .5
post_output = np.ones(len(actions), dtype = int) * .5

        
for t in np.arange(1,len(obs)):
    
    # Have the model produce behaviour 
    p_switch =  1 / float( 1 + np.exp(beta *((posterior) - 0.5)))
                
    ## Pick an action given the softmax p_switch 
    pOptions = [1-p_switch, p_switch]
    stay_switch = np.where(np.cumsum(pOptions) >= np.random.rand(1))[0][0]
    # 0: stay, 1: switch     
    actions[t]  = actions[t-1] if stay_switch == 0 else int(1 - actions[t-1])
    if actions[t] == true_state[t]: 
        obs[t] = np.random.binomial(1,pWin)
    else: 
        obs[t] = np.random.binomial(1,1-pWin)
            
        
    # Compute prior of trial t conditioned on actionmu_ 
    if actions[t] == actions[t-1]: # If stay action:
        prior = A_stay[0,0] * posterior + A_stay[0,1] * (1-posterior)
    elif actions[t] != actions[t-1]: # If switch action:
        prior = A_switch[0,0] * posterior + A_switch[0,1] * (1-posterior)
    # Compute posterior of trial t
    # posterior = prior * B_correct[obs[t]] / (B_correct[obs[t]] * prior + B_incorrect[obs[t]] * (1-prior))
    B_incorrect = 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-1/2 * ((obs[t] - mu_0)/sigma)**2)        
    B_correct = 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-1/2 * ((obs[t] - mu_1)/sigma)**2)
    posterior = prior * B_correct / (B_correct * prior + B_incorrect * (1-prior))
    
    # output: 0 means left choice, 1 means right choice    
    post_output[t] = (posterior)
    

    
    
output = np.array(post_output) # belief in 'correct' state 

inferred_state = np.ones(len(true_state))
inferred_state[np.where(output >= 0.5)[0]] = actions[np.where(output >= 0.5)[0]]
inferred_state[np.where(output < 0.5)[0]] = 1 - actions[np.where(output < 0.5)[0]]

%matplotlib qt5
plt.plot(true_state); 
plt.scatter(np.arange(200), actions, alpha=0.2); 

plt.plot(output, c='red', alpha=0.5);
plt.scatter(np.where(obs)[0], actions[np.where(obs)], alpha=0.6, color='purple');
plt.plot(inferred_state, alpha=.6)




