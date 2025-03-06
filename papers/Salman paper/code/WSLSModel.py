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


pytensor.config.floatX = 'float32'
class WSLSModel(object):
    """
    Win-stay-lose-shift model. An alternative to reward computation and tracking a la RW. 
    Only free parametera will be epsilon and beta. 
    """

    def __init__(self) -> None:
        self.model_params = ['bias'] # subject-level characteristics 

    @staticmethod
    def p_from_prev_WSLS(prev_actions, prev_rewards, P, bias):

        # Update the probability of the previous action to happen again
        P = pt.set_subtensor(P[prev_actions], (prev_rewards * (1-bias/2)) + ((1-prev_rewards) * (bias/2)))
        P = pt.set_subtensor(P[1-prev_actions], (prev_rewards * (bias / 2)) + ((1-prev_rewards) * (1-bias/2)))
       
        return P 

        
    def right_action_probs(self, actions=None, rewards=None, **kwargs):
        """

        """
        
        for p in self.model_params:
            if p not in kwargs:
                raise ValueError('Must provide parameter %s.' % p)
        
        bias = kwargs['bias']

        prev_actions = pt.as_tensor_variable(actions[:-1], dtype="int32")
        prev_rewards =  pt.as_tensor_variable(rewards[:-1], dtype="int32")

        P = 0.5 * pt.ones((2), dtype='float32')

        P, _ = pytensor.scan(
            fn=self.p_from_prev_WSLS,
            sequences=[prev_actions, prev_rewards],
            outputs_info=[P],
            non_sequences=[bias]
        )

        return P[:, 1]

    def make_model(self, actions=None, rewards=None):
        """
        Sample to infer posterior for parameters 
        """

        # actions_ = aesara.shared(np.asarray(actions, dtype='int16'))
        with pm.Model() as m:

            # Priors    
            bias = pm.Beta('bias', 1, 1.5, testval=0.1)

            action_probs = self.right_action_probs(actions=actions, rewards=rewards, bias=bias)
            like = pm.Bernoulli('like', p=action_probs, observed=actions[1:])

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

        bias = params['bias']
 
         # first trial: choose randomly           
        if n_trials==60:
            reversal_trials = [13, 24, 35, 48] # these are the first trials of the new block 
        elif n_trials==80: 
            reversal_trials = [16, 33, 48, 65]        
        a = [np.random.choice(2, 1, p=[0.5, 0.5])[0]] # choice acc. to choice probababilities [b, 1-b]
        r = [(np.random.randint(0,2) < mu[a[0]])]    # reward based on choice
    
        for t in range(n_trials-1):
            if t in reversal_trials: 
                mu = mu[::-1]
            # choice depends on last reward
            if r[t]==1:  # win stay (with probability 1-epsilon)
                p = bias / 2*np.ones(2)
                p[a[t]] = 1-bias/2
            elif r[t]==0: # lose shift (with probability 1-epsilon)
                p = (1-bias/2) * np.ones(2)
                p[a[t]] = bias/2

            a.append(np.random.choice(2, 1, p=p)[0])  # choice acc. to choice probababilities
            r.append((np.random.uniform() < mu[a[-1]]))     # reward based on choice
                        
        return np.squeeze(a), np.squeeze(r)