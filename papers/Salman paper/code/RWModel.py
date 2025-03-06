import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import pymc as pm
from pytensor import tensor as pt 
import pytensor
import arviz as az
from model_utils import DivergenceCounter
# Setting float precision in pytensor
pytensor.config.floatX = "float32"

from jax import config

import numpyro

numpyro.set_host_device_count(4)

config.update("jax_enable_x64", False)

class RWModel(object):
    """
    Classical RescorlaWagner Model. Fit using Bayesian inference. Add on to this class to implement other RL models.
    """

    def __init__(self) -> None:
        self.model_params = ['alpha', 'beta'] # subject-level characteristics 

    def update_Q(self, actions=None, rewards=None, Qs=None, alpha=None):
        """
        This function updates the Q table according to the RL update rule.
        It will be called by theano.scan to do so recursevely, given the observed data and the alpha parameter
        """

        Qs = pt.set_subtensor(Qs[actions], Qs[actions] + alpha * (rewards - Qs[actions]))

        return Qs

    def right_action_probs(self, actions=None, rewards=None,  **kwargs):
        """
        Compute the probability of choosing the right option. This probability will be used to parameterize the Bernoulli distribution 
        characterizing action selection.
        """
        
        for p in self.model_params:
            if p not in kwargs:
                raise ValueError('Must provide parameter %s.' % p)
        
        # Unpack the parameters
        alpha = kwargs['alpha']
        beta = kwargs['beta']

        # Convert the data to aesara shared variables
        rewards = pt.as_tensor_variable(rewards, dtype="int32")
        actions = pt.as_tensor_variable(actions, dtype="int32")

        # Compute the Qs values using a aesara scan (aka a loop in theano)
        Qs = 0.5 * pt.ones((2,), dtype="float64")
        Qs, updates = pytensor.scan(
            fn=self.update_Q,
            sequences=[actions, rewards],
            outputs_info=[Qs],
            non_sequences=[alpha])

        # shape: (n_trials,  prev_choice): Take all but the last trial 

        # Apply the softmax transformation
        Qs = Qs[:-1] * beta
        log_prob_actions = Qs - pt.logsumexp(Qs, axis=1, keepdims=True)

        # Return the probabilities for the right action, in the original scale
        return pt.exp(log_prob_actions[:, 1]) 

    # def theano_llik_td(self, actions=None, rewards=None, **kwargs):
    #     """
    #     Compute the log-likelihood of the data given the parameters.

    #     """
    #     for p in self.model_params:
    #         if p not in kwargs:
    #             raise ValueError('Must provide parameter %s.' % p)
        
    #     alpha = kwargs['alpha']
    #     beta = kwargs['beta']

    #     rewards = theano.shared(np.asarray(rewards, dtype='int16'))
    #     actions = theano.shared(np.asarray(actions, dtype='int16'))

    #     # Compute the Qs values
    #     Qs = 0.5 * tt.ones((2), dtype='float64')
    #     Qs, updates = theano.scan(
    #         fn=self.update_Q,
    #         sequences=[actions, rewards],
    #         outputs_info=[Qs],
    #         non_sequences=[alpha])

    #     # Apply the sotfmax transformation
    #     Qs_ = Qs[:-1] * beta
    #     log_prob_actions = Qs_ - pm.math.logsumexp(Qs_, axis=1)

    #     # Calculate the negative log likelihod of the observed actions
    #     log_prob_actions = log_prob_actions[tt.arange(actions.shape[0]-1), actions[1:]]
    #     return tt.sum(log_prob_actions)  # PyMC makes it negative by default

    def make_model(self, actions=None, rewards=None):
        """
        Make the model in PYMC. 
        """

        # actions_ = aesara.shared(np.asarray(actions, dtype='int16'))
        with pm.Model() as model:

            alpha = pm.Beta('alpha', 1.2, 1.2)
            beta = pm.HalfNormal('beta', 5)

            action_probs = self.right_action_probs(actions=actions, rewards=rewards, alpha=alpha, beta=beta)
            like = pm.Bernoulli('like', p=action_probs, observed=actions[1:])

        return model

    def fit(self, model, tune=500, target_accept=0.8, cores=1):
        """
        Sample to infer posterior for parameters.
        """

        with model:
            callback = DivergenceCounter()
            idata = pm.sample(progressbar=False, return_inferencedata=True, tune=tune, target_accept=target_accept, cores=cores, callback=callback)
            # nuts_sampler="numpyro", 
            idata = pm.compute_log_likelihood(idata)
            # map, opt_result = pm.find_MAP(model=model, return_raw=True) 
        
        return idata, callback

    # def llk_MLE(self, pars, actions=None, rewards=None):
        
    #     alpha, beta = pars 

    #     Q = np.array([.5,.5])
    #     choiceP = []
        
    #     for t in range(len(actions)):

    #         # compute choice probabilities
    #         p = np.exp(beta*Q) / np.sum(np.exp(beta*Q))
            
    #         choiceP.append(p[actions[t]])

    #         # update values
    #         delta = rewards[t] - Q[actions[t]]
    #         Q[actions[t]] = Q[actions[t]] + alpha * delta

    #     # return negative log-likelihood
    #     return - np.sum(np.log(np.array(choiceP)+10**-5))

    # def fit_MLE(self, actions=None, rewards=None):
    #     """
    #     Fit the model using MLE instead of Bayesian inference.
    #     """

    #     x0 = [np.random.uniform(), np.random.exponential()]
    #     bounds = [(0,1), (0,20)]

    #     res = scipy.optimize.minimize(self.llk_MLE, args = (actions, rewards), method='L-BFGS-B', x0=x0, bounds=bounds)
        
    #     return len(x0)* np.log(len(actions)) + 2*res.fun, res.x, -res.fun # BIC, pars, nLL

    # def fit(self, actions=None, rewards=None, method='Bayes'):
    #     """
    #     """
    #     if method == 'MLE':
    #         return self.fit_MLE(actions, rewards)
    #     elif method == 'Bayes':
    #         return self.fit_Bayes(actions, rewards)

    def compute_posterior_mean_param(self, idata=None):
        """
        extract the parameters from the posterior
        """
        model_dict = {f'{param_name}':[] for param_name in self.model_params}

        for p in self.model_params:
            model_dict[p].append(idata.posterior[p].mean().values)

        return model_dict

    def simulate(self, n_trials=60, mu=[.80,.20], params=None, **kwargs):
        """
        Simulate a sequence of actions and rewards. OR, alternatively, use the subject's actions to estimate latent parameters.
        """

        
        alpha = params['alpha']
        beta = params['beta']

        if ('actions' in kwargs) & ('rewards' in kwargs):
            actions = kwargs['actions']
            rewards = kwargs['rewards']
            sim_flag = False
        else:
            # simulate actions and rewards 
            actions = np.zeros(n_trials, dtype="int")
            rewards = np.zeros(n_trials, dtype="int")
            sim_flag = True
            
        # initialise Q values for each of the two options
        Qs = np.zeros((n_trials, 2))
        rpe = np.zeros(n_trials)

        # Initialize Q table
        Q = np.array([0.5, 0.5])
        if sim_flag:    
            # loop over trials
            if n_trials==60:
                reversal_trials = [13, 24, 35, 48] # these are the first trials of the new block 
            elif n_trials==80: 
                reversal_trials = [16, 33, 48, 65]
            for t in range(n_trials):
                if t in reversal_trials: 
                    mu = mu[::-1]
                # Apply the Softmax transformation
                exp_Q = np.exp(beta * Q)
                prob_a = exp_Q / np.sum(exp_Q)

                # Simulate choice and reward
                a = np.random.choice([0, 1], p=prob_a)
                r = np.random.random() < mu[a]

                rpe = r - Q[a]
                # Update Q table
                Q[a] = Q[a] + alpha * rpe

                # Store values
                actions[t] = a
                rewards[t] = r
                Qs[t] = Q.copy()

            return actions, rewards
        else:
            for t in range(n_trials):
                # Update Q table
                rpe[t] = rewards[t] - Q[int(actions[t])]
                Q[int(actions[t])] = Q[int(actions[t])] + alpha * (rpe[t])

            return rpe