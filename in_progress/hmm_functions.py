#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 14:42:01 2021

@author: vman
"""

#@title Helper functions


    
def forgetting(p0, A):
  """Calculate the forward predictive distribution in a discrete Markov chain

  Args:
    p0 (numpy vector): a discrete probability vector
    A (numpy matrix): the transition matrix, A[i,j] means the prob. to
    switch FROM i TO j

  Returns:
    p1 (numpy vector): the predictive probabilities in next time step
  """
  ############################################################################
  # Insert your code here to:
  #      Compute the marginal distribution of Markov chain in next time step
  #      Hint: use matrix multiply and be careful about the index orders
  # raise NotImplementedError("function `markov_forward` incomplete")
  ############################################################################
  p1 = A.T @ p0
  return p1
  

# Forward algorithm
def one_step_update(model, posterior_tm1, Y_t):
  """Given a HMM model, calculate the one-time-step updates to the posterior.
  Args:
    model (GaussianHMM instance): the HMM
    posterior_tm1 (numpy array): Posterior at `t-1`
    Y_t (numpy array): Observation at `t`
    Returns:
    posterior_t (numpy array): Posterior at `t`
  """
  prediction = model.transmat_ @ posterior_tm1
  likelihood = np.exp(model._compute_log_likelihood(Y_t))
  posterior_t = prediction * likelihood
  return posterior_t


np.random.seed(101)
switch_prob = 0.1
noise_level = 0.5
nsample = 50
T = 160
model = create_model(switch_prob, noise_level)

posterior_list = []
for i in range(nsample):
  predictive_probs, posterior_probs = simulate_forward_inference(model, T)
  posterior_list.append(posterior_probs)
posterior_matrix = np.asarray(posterior_list)

with plt.xkcd():
  plot_evidence_vs_noevidence(posterior_matrix, predictive_probs)
  
  
def simulate_forward_inference(model, T, data=None):
  """
  Given HMM `model`, calculate posterior marginal predictions of x_t for T-1 time steps ahead based on
  evidence `data`. If `data` is not give, generate a sequence of observations from first component.

  Args:
    model (GaussianHMM instance): the HMM
    T (int): length of returned array

  Returns:
    predictive_state1: predictive probabilities in first state w.r.t no evidence
    posterior_state1: posterior probabilities in first state w.r.t evidence
  """

  # First re-calculate hte predictive probabilities without evidence
  predictive_probs = simulate_prediction_only(model, T)
  # Generate an observation trajectory condtioned on that latent state x is always 1
  if data is not None:
    Y = data
  else:
    Y = np.asarray([model._generate_sample_from_state(0) for _ in range(T)])
  # Calculate marginal for each latent state x_t
  pt = np.exp(model._compute_log_likelihood(Y[[0]])) * model.startprob_
  pt /= np.sum(pt)
  posterior_probs = np.zeros((T, pt.size))
  posterior_probs[0] = pt

  for t in range(1, T):
    posterior = one_step_update(model, posterior_probs[t-1], Y[[t]])
    # normalize and add to the list
    posterior /= np.sum(posterior)
    posterior_probs[t] = posterior
  posterior_state1 = np.asarray([p[0] for p in posterior_probs])
  predictive_state1 = np.asarray([p[0] for p in predictive_probs])
  return predictive_state1, posterior_state1


# Infer state sequence
predictive_probs, posterior_probs = simulate_forward_inference(model, nstep,
                                                               observations)
states_inferred = posterior_probs <= 0.5

## Check if you get same results with viterbi...? 






                    
                    
    
        
    def forward(self, obs, actions): 
        alpha = np.empty((self.numStates, len(obs)), dtype = float)
        # Initialize forward cache
        alpha[:,0] = self.pi * self.B_means 
        # Induction
        for t in np.arange(1,len(obs)):
            for j in np.arange(self.numStates):                
                if actions[t] == actions[t-1]: # If stay action:
                    alpha[j,t] = np.sum(alpha[i, t-1] * self.A_stay[i, j] * self.B_means[j] for i in np.arange(self.numStates))
                elif actions[t] != actions[t-1]: # If switch action:
                    alpha[j,t] = np.sum(alpha[i, t-1] * self.A_switch[i, j] * self.B_means[j] for i in np.arange(self.numStates))                
        # Termination 
        likelihood = np.sum(alpha[:,-1])
        return likelihood
        
        

    
    

    def backward(self, obs):
        self.bwk = [{} for t in range(len(obs))]
        T = len(obs)
        # Initialize base cases (t == T)
        for y in self.states:
            self.bwk[T-1][y] = 1 #self.A[y]["Final"] #self.pi[y] * self.B[y][obs[0]]
        for t in reversed(range(T-1)):
            for y in self.states:
                self.bwk[t][y] = sum((self.bwk[t+1][y1] * self.A[y][y1] * self.B[y1][obs[t+1]]) for y1 in self.states)
        prob = sum((self.pi[y]* self.B[y][obs[0]] * self.bwk[0][y]) for y in self.states)
        return prob

    def forward(self, obs):
        self.fwd = [{}]     
        # Initialize base cases (t == 0)
        for y in self.states:
            self.fwd[0][y] = self.pi[y] * self.B[y][obs[0]]
        # Run Forward algorithm for t > 0
        for t in range(1, len(obs)):
            self.fwd.append({})     
            for y in self.states:
                self.fwd[t][y] = sum((self.fwd[t-1][y0] * self.A[y0][y] * self.B[y][obs[t]]) for y0 in self.states)
        prob = sum((self.fwd[len(obs) - 1][s]) for s in self.states)
        return prob

    def viterbi(self, obs):
        vit = [{}]
        path = {}     
        # Initialize base cases (t == 0)
        for y in self.states:
            vit[0][y] = self.pi[y] * self.B[y][obs[0]]
            path[y] = [y]
     
        # Run Viterbi for t > 0
        for t in range(1, len(obs)):
            vit.append({})
            newpath = {}     
            for y in self.states:
                (prob, state) = max((vit[t-1][y0] * self.A[y0][y] * self.B[y][obs[t]], y0) for y0 in self.states)
                vit[t][y] = prob
                newpath[y] = path[state] + [y]     
            # Don't need to remember the old paths
            path = newpath
        n = 0           # if only one element is observed max is sought in the initialization values
        if len(obs)!=1:
            n = t
        (prob, state) = max((vit[n][y], y) for y in self.states)
        return (prob, path[state])

    def forward_backward(self, obs): # returns model given the initial model and observations        
        gamma = [{} for t in range(len(obs))] # this is needed to keep track of finding a state i at a time t for all i and all t
        zi = [{} for t in range(len(obs) - 1)]  # this is needed to keep track of finding a state i at a time t and j at a time (t+1) for all i and all j and all t
        # get alpha and beta tables computes
        p_obs = self.forward(obs)
        self.backward(obs)
        # compute gamma values
        for t in range(len(obs)):
            for y in self.states:
                gamma[t][y] = (self.fwd[t][y] * self.bwk[t][y]) / p_obs
                if t == 0:
                    self.pi[y] = gamma[t][y]
                #compute zi values up to T - 1
                if t == len(obs) - 1:
                    continue
                zi[t][y] = {}
                for y1 in self.states:
                    zi[t][y][y1] = self.fwd[t][y] * self.A[y][y1] * self.B[y1][obs[t + 1]] * self.bwk[t + 1][y1] / p_obs
        # now that we have gamma and zi let us re-estimate
        for y in self.states:
            for y1 in self.states:
                # we will now compute new a_ij
                val = sum([zi[t][y][y1] for t in range(len(obs) - 1)]) #
                val /= sum([gamma[t][y] for t in range(len(obs) - 1)])
                self.A[y][y1] = val
        # re estimate gamma
        for y in self.states:
            for k in self.
            
            : # for all symbols vk
                val = 0.0
                for t in range(len(obs)):
                    if obs[t] == k :
                        val += gamma[t][y]                 
                val /= sum([gamma[t][y] for t in range(len(obs))])
                self.B[y][k] = val
        return
    
    


