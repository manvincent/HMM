

# Import modules
import numpy as np
from itertools import compress
from scipy.stats import gamma, beta, invgamma
# Define model types
class ModelType(object):

    def __init__(self, payOut, emission_type = 'gaussian'):
        '''
        Baseline HMM conditioned on state and behaviour (stay/switch)
        Implementation of HMM described in Hampton et al (2006) J Neurosci
        Gaussian emission distribution

        Parameters
        ----------
        payOut : float array
            Vector of outcomes from the data
        emission_type : string, optional
            Options: 'discrete' or 'gaussian'
            Toggles between discrete emission probabilities vs. Gaussian
            emissions.
            The default is = 'discrete'.

        Returns
        -------
        None.

        '''
        self.numParams = 5
        self.emission_type = emission_type
        # Define parameter lower and upper bounds
        self.numStates = 2
        self.delta_bounds = (0.1,0.9)
        self.smBeta_bounds = (1, 20)
        self.mu_0_bounds = (np.min(payOut), 0) # parameter of p(out | state = incorrect)
        self.mu_1_bounds = (0, np.max(payOut)) # parameter of p(out | state = correct)
        self.sigma_bounds = (0.1, 3)

        # Define prior distributions hyperparameters
        self.delta_a = self.delta_b = 2 # Beta distr. prior over delta
        self.smBeta_shape = 10 # Gamma distr. prior over smBeta
        self.smBeta_loc = 0
        self.smBeta_scale = 1.2
        self.sigma_shape = 2
        self.sigma_loc = 0
        self.sigma_scale = 1
        return

    def genPriors(self, arr):
        '''
        Generate the grid of parameter values (in parameter recovery or to
                                               initialize gradient descent)
        Parameter values are generated according to each parameter's
        respective prior distribution.

        Parameters
        ----------
        arr : int
            Defines grid size.

        Returns
        -------
        genVal : dict
            Dictionary of initialized parameter values along prior-weighted
            grid.

        '''

        # Distributions
        delta_genDistr = beta(self.delta_a, self.delta_b)
        smBeta_genDistr = gamma(self.smBeta_shape, self.smBeta_loc, self.smBeta_scale)
        sigma_genDistr = invgamma(self.sigma_shape, self.sigma_loc, self.sigma_scale)

        # CDF at boundaries
        delta_cdf = delta_genDistr.cdf(self.delta_bounds)
        smBeta_cdf = smBeta_genDistr.cdf(self.smBeta_bounds)
        sigma_cdf = sigma_genDistr.cdf(self.sigma_bounds)

        # Generative (ground truth) parameter values using ppf
        delta_genVal = delta_genDistr.ppf(np.linspace(*delta_cdf, num=arr))
        smBeta_genVal = smBeta_genDistr.ppf(np.linspace(*smBeta_cdf, num=arr))
        sigma_genVal = sigma_genDistr.ppf(np.linspace(*sigma_cdf, num=arr))
        mu_0_genVal = np.linspace(*self.mu_0_bounds, arr)
        mu_1_genVal  = np.linspace(*self.mu_1_bounds, arr)

        # Pack up generative parameter values
        genVal = dict(delta = delta_genVal,
                      smBeta = smBeta_genVal,
                      mu_0 = mu_0_genVal,
                      mu_1 = mu_1_genVal,
                      sigma = sigma_genVal)
        return genVal

    def paramPriors(self):
        '''
        Creates log pdfs for each parameter's prior distribution for MAP.
        Excludes parameters with uniform priors (MAP = MLE).

        Returns
        -------
        delta_logpdf : function
            Beta distribution log pdf for the transition probability
        smBeta_logpdf : function
            Gamma distribution log pdf for the softmax beta

        '''
        # Prior distribution for transition probability
        delta_logpdf = lambda x: np.sum(np.log(beta.pdf(x, self.delta_a, self.delta_b)))
        delta_logpdf.__name__ = "A_logpdf"
        # Prior distribution of softmax beta
        smBeta_logpdf = lambda x: np.sum(np.log(gamma.pdf(x, self.smBeta_shape, self.smBeta_loc, self.smBeta_scale)))
        smBeta_logpdf.__name__ = "smB_logpdf"
        # Prior distribution for gaussian sigma
        sigma_logpdf = lambda x: np.sum(np.log(invgamma.pdf(x, self.sigma_shape, self.sigma_loc, self.sigma_scale)))
        sigma_logpdf.__name__ = "sigma_logpdf"
        return dict(delta_prior = delta_logpdf,
                    smBeta_prior = smBeta_logpdf,
                    sigma_prior = sigma_logpdf)

    def belief_propogation(self, posterior, delta, mu_0, mu_1, sigma, obs, actions):
        '''
        Core hidden markov model function which updates the model on one trial.
        Compute current trial's prior state belief and updates according to
        action (stay/switch) and observed outcome.

        Parameters
        ----------
        posterior : float
            Posterior probability of correct action on previous trial p(X_{t-1} =  correct)
        delta : float
            Transition probability free parameter.
        mu_0 : float
            Mean of Gaussian distribution governing p(outcome | state = incorrect).
            If discrete emission probabilities, equals p(loss | state = correct).
        mu_1 : float
            Mean of Gaussian distribution governing p(outcome | state = correct).
            If discrete emission probabilities, equals p(win | state = correct).
        sigma : float
            Std. dev. of Gaussian distribution governing p(outcome | state)
        obs : float
            Outcome of current trial.
        actions : tuple
            Tuple containing the action chosen on t-1 and t

        Returns
        -------
        prior: float
            Prior probability of correct action on current trial p(X_{t} = correct)
        posterior: float
            Posterior probability of correct action on current trial

        '''

        # Transition matrix is symmetrical so can be reduced to a vector for stay[0] vs switch[1]
        A = np.array([1-delta, delta])

        # Update t-1 posterior to t prior via transition probability
        if len(np.unique(actions)) == 1: # If stay
            prior = A[0] * posterior + A[1] * (1-posterior)
        else: # If switch
            prior = A[1] * posterior + A[0] * (1-posterior)

        # Compute t posterior given p(out | state)
        if self.emission_type == 'discrete':
            # Discrete Emission probabilities
            B_correct = np.array([mu_0, mu_1])
            B_incorrect = -np.flip(B_correct)
            obs_idx = int(obs > 0) # Recode win/loss as 1/0 int for indexing
            posterior = prior * B_correct[obs_idx] / (B_correct[obs_idx] * prior + B_incorrect[obs_idx] * (1-prior))
        elif self.emission_type == 'gaussian':
            # Gaussian Emission probabilities
            B_incorrect = 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-1/2 * ((obs - mu_0)/sigma)**2)
            B_correct = 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-1/2 * ((obs - mu_1)/sigma)**2)
            posterior = prior * B_correct / (B_correct * prior + B_incorrect * (1-prior))
        return posterior

    def actor(self, posterior, smBeta, prev_action):
        # Action selection through logistic function
        p_switch =  1 / float( 1 + np.exp(smBeta *(posterior - 0.5)))
        ## Pick an action given the softmax p_switch
        pOptions = [1-p_switch, p_switch]
        stay_switch = np.where(np.cumsum(pOptions) >= np.random.rand(1))[0][0]
        # 0: stay, 1: switch
        curr_action = prev_action if stay_switch == 0 else int(1 - prev_action)
        # output: 0 means left choice, 1 means right choice
        return curr_action, pOptions


# Set up a container for model parameter estimates (fits)
class fitParamContain():
    def __init__(self, fitlikelihood):
        self.fitlikelihood = fitlikelihood
        return

    def action_HMM_fitParams(self, fitParams):
        self.delta = fitParams[0]
        self.smBeta = fitParams[1]
        self.mu_0 = fitParams[2]
        self.mu_1 = fitParams[3]
        self.sigma = fitParams[4]
        return self
