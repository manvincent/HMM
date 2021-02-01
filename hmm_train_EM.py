#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 12:50:50 2020

@author: vman
"""

def e_step(Y, psi, A, B):
  """Calculate the E-step (forward-backward) for HMM.

  Args:
    Y :             data (T)
    psi :           initial probabilities for each state (N states)
    A :             transition matrix, A[i,j] represents the prob to
                        switch from i to j. (N, N)
    B :             emission probabilities.(1,N)    
  
  Output:
    ll :             data log likelihood
    gamma :          singleton marginal distribution. (T, N)
    xi :             pairwise marginal distribution for adjacent nodes. 
                    (T-1, N, N)
  """
  n_trials = Y.shape[0]
  T = Y.shape[1]
  K = psi.size
  log_a = np.zeros((n_trials, T, K))
  log_b = np.zeros((n_trials, T, K))

  log_A = np.log(A)
  log_obs = stats.poisson(L * dt).logpmf(Y[..., None]).sum(-2)  # n_trials, T, K

  # forward pass
  log_a[:, 0] = log_obs[:, 0] + np.log(psi)
  for t in range(1, T):
    tmp = log_A + log_a[:, t - 1, : ,None]  # (n_trials, K,K)
    maxtmp = tmp.max(-2)  # (n_trials,K)
    log_a[:, t] = (log_obs[:, t] + maxtmp +
                    np.log(np.exp(tmp - maxtmp[:, None]).sum(-2)))

  # backward pass
  for t in range(T - 2, -1, -1):
    tmp = log_A + log_b[:, t + 1, None] + log_obs[:, t + 1, None]
    maxtmp = tmp.max(-1)
    log_b[:, t] = maxtmp + np.log(np.exp(tmp - maxtmp[..., None]).sum(-1))

  # data log likelihood
  maxtmp = log_a[:, -1].max(-1)
  ll = np.log(np.exp(log_a[:, -1] - maxtmp[:, None]).sum(-1)) + maxtmp

  # singleton and pairwise marginal distributions
  gamma = np.exp(log_a + log_b - ll[:, None, None])
  xi = np.exp(log_a[:, :-1, :, None] + (log_obs + log_b)[:, 1:, None] +
              log_A - ll[:, None, None, None])

  return ll.mean() / T / dt, gamma, xi


def m_step(gamma, xi, dt):
  """Calculate the M-step updates for the HMM spiking model.
  Args:
    gamma ():       Number of epochs of EM to run
    xi (numpy 3d array): Tensor of recordings, has shape (n_trials, T, C)
    dt (float):         Duration of a time bin
  Returns:
    psi_new (numpy vector): Updated initial probabilities for each state
    A_new (numpy matrix):   Updated transition matrix, A[i,j] represents the
                            prob. to switch from j to i. Has shape (K,K)
    L_new (numpy matrix):   Updated Poisson rate parameter for different
                            cells. Has shape (C,K)
  """
  # Calculate and normalize the new initial probabilities, psi_new
  psi_new = gamma[:, 0].mean(axis=0)
  # Make sure the probabilities are normalized

  psi_new /= psi_new.sum()
  # Calculate new transition matrix
  A_new = xi.sum(axis=(0, 1)) / gamma[:, :-1].sum(axis=(0, 1))[:, np.newaxis]
  # Calculate new firing rates
  L_new = (np.swapaxes(Y, -1, -2) @ gamma).sum(axis=0) / gamma.sum(axis=(0, 1)) / dt
  return psi_new, A_new, L_new


def run_em(epochs, Y, psi, A, L, dt):
  """Run EM for the HMM spiking model.

  Args:
    epochs (int):       Number of epochs of EM to run
    Y (numpy 3d array): Tensor of recordings, has shape (n_trials, T, C)
    psi (numpy vector): Initial probabilities for each state
    A (numpy matrix):   Transition matrix, A[i,j] represents the prob to switch
                        from j to i. Has shape (K,K)
    L (numpy matrix):   Poisson rate parameter for different cells.
                        Has shape (C,K)
    dt (float):         Duration of a time bin

  Returns:
    save_vals (lists of floats): Data for later plotting
    lls (list of flots):         ll Before each EM step
    psi (numpy vector):          Estimated initial probabilities for each state
    A (numpy matrix):            Estimated transition matrix, A[i,j] represents
                                 the prob to switch from j to i. Has shape (K,K)
    L (numpy matrix):            Estimated Poisson rate parameter for different
                                 cells. Has shape (C,K)
  """
  save_vals = []
  lls = []
  for e in range(epochs):

      # Run E-step
      ll, gamma, xi = e_step(Y, psi, A, L, dt)
      lls.append(ll)  # log the data log likelihood for current cycle

      if e % print_every == 0: print(f'epoch: {e:3d}, ll = {ll}')  # log progress
      # Run M-step
      psi_new, A_new, L_new = m_step(gamma, xi, dt)

      """Booking keeping for later plotting
      Calculate the difference of parameters for later
      interpolation/extrapolation
      """
      dp, dA, dL = psi_new - psi, A_new - A, L_new - L
      # Calculate LLs and ECLLs for later plotting
      if e in plot_epochs:
          b_min = -min([np.min(psi[dp > 0] / dp[dp > 0]),
                        np.min(A[dA > 0] / dA[dA > 0]),
                        np.min(L[dL > 0] / dL[dL > 0])])
          b_max = -max([np.max(psi[dp < 0] / dp[dp < 0]),
                        np.max(A[dA < 0] / dA[dA < 0]),
                        np.max(L[dL < 0] / dL[dL < 0])])
          b_min = np.max([.99 * b_min, b_lims[0]])
          b_max = np.min([.99 * b_max, b_lims[1]])
          bs = np.linspace(b_min, b_max, num_plot_vals)
          bs = sorted(list(set(np.hstack((bs, [0, 1])))))
          bs = np.array(bs)
          lls_for_plot = []
          eclls_for_plot = []
          for i, b in enumerate(bs):
              ll = e_step(Y, psi + b * dp, A + b * dA, L + b * dL, dt)[0]
              lls_for_plot.append(ll)
              rate = (L + b * dL) * dt
              ecll = ((gamma[:, 0] @ np.log(psi + b * dp) +
                       (xi * np.log(A + b * dA)).sum(axis=(-1, -2, -3)) +
                       (gamma * stats.poisson(rate).logpmf(Y[..., np.newaxis]).sum(-2)
                       ).sum(axis=(-1, -2))).mean() / T / dt)
              eclls_for_plot.append(ecll)
              if b == 0:
                  diff_ll = ll - ecll
          lls_for_plot = np.array(lls_for_plot)
          eclls_for_plot = np.array(eclls_for_plot) + diff_ll
          save_vals.append((bs, lls_for_plot, eclls_for_plot))
      # return new parameter
      psi, A, L = psi_new, A_new, L_new

  ll = e_step(Y, psi, A, L, dt)[0]
  lls.append(ll)
  print(f'epoch: {epochs:3d}, ll = {ll}')
  return save_vals, lls, psi, A, L