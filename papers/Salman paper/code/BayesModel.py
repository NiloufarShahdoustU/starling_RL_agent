import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import pymc as pm
from pytensor import tensor as pt 
import pytensor
import arviz as az
from model_utils import DivergenceCounter
# # Setting float precision in pytensor
# pytensor.config.floatX = "float32"

# from jax.config import config

# import numpyro

# numpyro.set_host_device_count(4)

# config.update("jax_enable_x64", False)

class BayesModel(object):
    """
    Bayesian filter model. Computes likelihoods that an action is correct or incorrect as a function of 
    of inferred p(reward) for the current state. The probability of an action from these likelihoods - but take into
    account the p(switch) that a state switch occurred. 
    
     Here, we are interested in whether the participants' mental model of the world
    captures the probability of a reward for the right action, and the probability of a reversal.

    """

    def __init__(self) -> None:
        self.model_params = ['p_switch', 'p_reward', 'beta'] # subject-level characteristics':  'persev'

    @staticmethod
    def _get_likelihoods(actions, rewards, p_reward, p_noisy):
        """
        https://github.com/MariaEckstein/SLCN/blob/4fb5955c1142fcbd8ec80d7fccdf6b35dbfd1616/models/PSModelFunctions2.py
        """
        
        LC = (rewards * (actions * p_reward + (1 - actions) * p_noisy)) + ((1-rewards) * (actions * (1 - p_reward) + (1 - actions) * (1 - p_noisy)))
        LI = (rewards * (actions * p_noisy + (1 - actions) * p_reward)) + ((1-rewards) * (actions * (1 - p_noisy) + (1 - actions) * (1 - p_reward)))

        return LC, LI

    @staticmethod
    def _post_from_lik(LC, LI, p_switch, beta):

        counter = 0
        ps = []
        for lc, li, in zip(LC, LI):
            if counter == 0:
                P = 0.5

            # Estimation phase: hidden state p(action t- 1 is correct given outcome t-1)
            P = (P * lc) / ((lc * P) + (li * (1-P)))

            # Prediction phase p(action t is correct)
            # Take into account that a switch might occur
            P = ((1-p_switch) * P) + (p_switch * (1-P))

            P = 1 / (1 + np.exp(beta * (0.5 - P)))
            P = 0.0001 + 0.9998 * P
            ps.append(P)
            counter += 1

        return pt.as_tensor_variable(ps[:-1], dtype="float64")

        # P = (P * LC) / ((LC * P) + (LI * (1-P)))
        # # Take into account that a switch might occur
        # P = ((1-p_switch) * P) + (p_switch * (1-P))

        # P = 1 / (1 + np.exp(beta * (0.5 - P)))
        # P = 0.0001 + 0.9998 * P

        # return P

    def right_action_probs(self, actions=None, rewards=None,  **kwargs):
        """
        """
        for p in self.model_params:
            if p not in kwargs:
                raise ValueError('Must provide parameter %s.' % p)

        p_switch = kwargs['p_switch']
        p_reward = kwargs['p_reward']
        p_noisy = kwargs['p_noisy']
        beta = kwargs['beta']
        
        rewards = pt.as_tensor_variable(rewards, dtype="int32")
        actions = pt.as_tensor_variable(actions, dtype="int32")

        LC, LI = self._get_likelihoods(actions, rewards, p_reward, p_noisy)

        # Get posterior & calculate probability of subsequent trial
        # P = pt.as_tensor_variable(0.5*np.ones((2)), name="P")

        # P = 0.5 * pt.ones((1,), dtype="float64")
        P =  self._post_from_lik(LC, LI, p_switch, beta)
        # P, _ = pytensor.scan(fn=self._post_from_lik,  
        #                                 sequences=[LC, LI],
        #                                 outputs_info=[P],
        #                                 non_sequences=[p_switch, beta])

        # shape of P: (n_trials, 2)
        return P
    

    def make_model(self, actions=None, rewards=None):
        """
        Sample to infer posterior for parameters 
        """

        actions_ = pt.as_tensor_variable(actions, dtype="int32")
        with pm.Model() as m:

            beta = pm.HalfNormal('beta', 7)
            # beta = pm.HalfNormal('beta', 7)
            # Values come from here: https://github.com/MariaEckstein/SLCN/blob/master/models/PSAllModels.py

            # 11/16/2022: I wonder if the reason this model fails so badly is because the fucking free parameters should be allowed to vary dynamically per trial?

            p_switch = pm.Beta('p_switch', 1, 4, initval=0.066)
            p_reward = pm.Beta('p_reward', 1.1, 1.1, initval=0.8)

            p_noisy = pm.Deterministic('p_noisy', 1e-5 * pt.ones(1, dtype='float64'))
            # p_reward = pm.Deterministic('p_reward', 0.8* tt.ones((1), dtype='float64'))

            P = self.right_action_probs(actions=actions, 
            rewards=rewards, 
            p_switch=p_switch, 
            p_reward=p_reward, 
            beta=beta,
            p_noisy=p_noisy)

            like = pm.Bernoulli('like', p=P, observed=actions_[1:]) 
            # predict from trial 3 on; discard last p_right because there is no trial to predict after the last value update
        return m 
    
    def fit(self, model, tune=500, target_accept=0.8, cores=1):
        """
        Sample to infer posterior for parameters.
        """

        with model:
            callback = DivergenceCounter()
            idata = pm.sample(return_inferencedata=True, tune=tune, target_accept=target_accept, cores=cores, callback=callback)
            # nuts_sampler="numpyro", progressbar=False, 
            idata = pm.compute_log_likelihood(idata)
            # map, opt_result = pm.find_MAP(model=model, return_raw=True) 
        
        return idata, callback

    def compute_posterior_mean_param(self, idata=None):
        """
        extract the parameters from the posterior
        """
        model_dict = {f'{param_name}':[] for param_name in self.model_params}

        for p in self.model_params:
            model_dict[p].append(idata.posterior[p].mean().values)

        return model_dict

    def simulate(self, n_trials=60, mu=[0.8, 0.2], params=None):
        # T: num of trials
        # mu: true reward probabilities
        # alpha: learning rate
        # beta: softmax inverse temperature

        def _post_from_lik_sim(lik_cor, lik_inc,
                        p_r, beta,
                        p_switch, verbose=False):

            if verbose:
                print('old p_r:\n{0}'.format(p_r.round(3)))

            # Apply Bayes rule: Posterior prob. that right action is correct, based on likelihood (i.e., received feedback)
            p_r = lik_cor * p_r / (lik_cor * p_r + lik_inc * (1 - p_r))

            # Take into account that a switch might occur
            p_r = (1 - p_switch) * p_r + p_switch * (1 - p_r)

            bpe = 1-p_r 
            # Bayesian predictive error: https://www.pnas.org/doi/10.1073/pnas.2212252120#sec-3
            # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9108470/
            

            # Log-transform probabilities
            p_right = 1 / (1 + np.exp(beta * (0.5-p_r)))
            # p_right = 1 / (1 + np.exp(-beta * (p_right0 - (1 - p_right0))))
            p_right = 0.0001 + 0.9998 * p_right  # make 0.0001 < p_right < 0.9999

            # p_r is the actual probability of right, which is the prior for the next trial
            # p_right is p_r after adding perseveration and log-transform, used to select actions
            return p_r, p_right
            # , bpe
            #, p_right

        p_switch = params['p_switch']
        p_reward = params['p_reward']
        p_noisy = params['p_noisy']
        beta = params['beta']

        ps_right = np.zeros(n_trials)
        lik_cors = []
        lik_incs = []
        p_rs = []
        p_rights = []

        actions = np.zeros(n_trials, dtype="int")
        rewards = np.zeros(n_trials, dtype="int")
        bpes = np.zeros(n_trials, dtype="int")
        # loop over trials
        # special condition for my task design: 
        if n_trials==60:
            reversal_trials = [13, 24, 35, 48] # these are the first trials of the new block 
        else: 
            reversal_trials = [16, 33, 48, 65]
        for t in range(n_trials):
            if t in reversal_trials: 
                mu = mu[::-1]
            if t == 0:
                p_r = 0.5 
                # p_right = 0.5 
            else:
                lik_cor, lik_inc = self._get_likelihoods(actions[t-1], rewards[t-1], p_reward, p_noisy)
                lik_cors.append(np.append([t], lik_cor))
                lik_incs.append(np.append([t], lik_inc))

                p_r, p_right = _post_from_lik_sim(lik_cor, lik_inc, p_r, beta, p_switch)
                p_rs.append(np.append([t], p_r))
                p_rights.append(np.append([t], p_right))

                a = np.random.choice([0, 1], p=[1-p_right, p_right])
                r = np.random.random() < mu[a]

                actions[t] = a
                rewards[t] = r
                # bpes[t] = bpe

        return actions, rewards
        #, bpes