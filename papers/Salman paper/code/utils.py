# Utility functions for image memorability ratings
import pandas as pd 
import numpy as np 
import scipy as sp 
import os 
from scipy.stats import norm, zscore, linregress
from scipy.stats import t, norm
from math import atanh, pow
from numpy import tanh

# Note: Much of the following is ported from: https://github.com/cvzoya/memorability-distinctiveness

def dprime(pHit, pFA, PresentT, AbsentT, criteria=False):
    """
    Note: from: http://nikos-konstantinou.blogspot.com/2010/02/dprime-function-in-matlab.html
    
    
    Parameters
    ----------
    pHit : float
        The proportion of "Hits": P(Yes|Signal)
    pFA : float
        The proportion of "False Alarms": P(Yes|Noise)
    PresentT : int
        The number of Signal Present Trials e.g. length(find(signal==1))
    AbsentT : int
        The number of Signal Absent Trials e.g. length(find(signal==0))

        
    Returns
    -------
    dPrime: float
        signal detection theory sensitivity measure 
    
    beta: float
        optional criterion value
        
    C: float
        optional criterion value
        
    """

    if pHit == 1: 
        # if 100% Hits
        pHit = 1 - (1/(2*PresentT))
    
    if pFA == 0: 
        # if 0% FA 
        pFA = 1/(2*AbsentT)
        
    # Convert to Z-scores
    
    zHit = norm.ppf(pHit) 
    zFA = norm.ppf(pFA) 
    
    # calculate d-prime 
    
    dPrime = zHit - zFA 
    
    if criteria:
        beta = np.exp((zFA**2 - zHit**2)/2)
        C = -0.5 * (zHit + zFA)    
        return dPrime, beta, C
    else:
        return dPrime

          
def matthews_corrcoef(tp, tn, fp, fn):
    """
    """
    
    numerator = (tp * tn) - (fp * fn)
    
    denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    
    if denominator == 0:
        return 0
    
    return numerator / denominator

    
# def calcMI(pmf):
#     """
    
#     Parameters
#     ----------
    
    
#     Returns
#     -------
    
#     """
        
#     pmf_1 = np.sum(pmf,axis=1) #  marginal over first variable
#     pmf_2 = np.sum(pmf,axis=0) # marginal over second variable
    
#     MI = 0
#     for i in range(np.shape(pmf)[0]):
#         for j in range(np.shape(pmf)[1]):
#             MI += pmf[i, j] * np.log(pmf[i, j] / (pmf_1[i]*pmf_2[j]))
            
#     return MI

def compute_memorability_scores(hits, false_alarms, misses, correct_rejections):
    """
    Parameters
    ----------
    hits : array-like
        TODO
    false_alarms : array-like
        TODO
    misses : array_like 
        TODO
    correct_rejections : array_like 
        TODO
        
    Returns
    -------
    memory_ratings : pandas DataFrame 
        DataFrame with the following ratings added: HR (hit rate), FAR (false alarm rate), ACC (accuracy), DPRIME (d-prime), MI (mutual information)
    """
    
    

    len_args = [len(hits), len(false_alarms), len(misses), len(correct_rejections)]
    if not all(len_args[0] == _arg for _arg in len_args[1:]):
            raise ValueError("All parameters must be the same length.")
    
    memory_ratings = pd.DataFrame(columns = ['HR', 'FAR', 'ACC', 'DPRIME'])
    # , 'MI'
    
    reg = 0.1 # regularization for MI calculation

    nstimuli = len(hits) 

    hm = hits+misses
    fc = false_alarms+correct_rejections

    hrs = hits/hm
    fars = false_alarms/fc
    accs = (hits+correct_rejections)/(hm+fc)

    dp = []
#     mis = []
    for i in range(nstimuli):
        dp.append(dprime(hrs[i], fars[i], hm[i], fc[i]))
#         pmf = np.array([[correct_rejections, misses], 
#                [false_alarms, hits]]) + reg
#         pmf = pmf/np.sum(pmf)
#         mis.append(calcMI(pmf))
    

    memory_ratings['HR'] = hrs
    memory_ratings['FAR'] = fars
    memory_ratings['ACC'] = accs
    memory_ratings['DPRIME'] = dp
#     memory_ratings['MI'] = mis
    
    return memory_ratings

"""
Functions for calculating the statistical significant differences between two dependent or independent correlation
coefficients.
The Fisher and Steiger method is adopted from the R package http://personality-project.org/r/html/paired.r.html
and is described in detail in the book 'Statistical Methods for Psychology'
The Zou method is adopted from http://seriousstats.wordpress.com/2012/02/05/comparing-correlations/
Credit goes to the authors of above mentioned packages!
Author: Philipp Singer (www.philippsinger.info)
"""

from __future__ import division

__author__ = 'psinger'

def rz_ci(r, n, conf_level = 0.95):
    zr_se = pow(1/(n - 3), .5)
    moe = norm.ppf(1 - (1 - conf_level)/float(2)) * zr_se
    zu = atanh(r) + moe
    zl = atanh(r) - moe
    return tanh((zl, zu))

def rho_rxy_rxz(rxy, rxz, ryz):
    num = (ryz-1/2.*rxy*rxz)*(1-pow(rxy,2)-pow(rxz,2)-pow(ryz,2))+pow(ryz,3)
    den = (1 - pow(rxy,2)) * (1 - pow(rxz,2))
    return num/float(den)

def dependent_corr(xy, xz, yz, n, twotailed=True, conf_level=0.95, method='steiger'):
    """
    Calculates the statistic significance between two dependent correlation coefficients
    @param xy: correlation coefficient between x and y
    @param xz: correlation coefficient between x and z
    @param yz: correlation coefficient between y and z
    @param n: number of elements in x, y and z
    @param twotailed: whether to calculate a one or two tailed test, only works for 'steiger' method
    @param conf_level: confidence level, only works for 'zou' method
    @param method: defines the method uses, 'steiger' or 'zou'
    @return: t and p-val
    """
    if method == 'steiger':
        d = xy - xz
        determin = 1 - xy * xy - xz * xz - yz * yz + 2 * xy * xz * yz
        av = (xy + xz)/2
        cube = (1 - yz) * (1 - yz) * (1 - yz)

        t2 = d * np.sqrt((n - 1) * (1 + yz)/(((2 * (n - 1)/(n - 3)) * determin + av * av * cube)))
        p = 1 - t.cdf(abs(t2), n - 3)

        if twotailed:
            p *= 2

        return t2, p
    elif method == 'zou':
        L1 = rz_ci(xy, n, conf_level=conf_level)[0]
        U1 = rz_ci(xy, n, conf_level=conf_level)[1]
        L2 = rz_ci(xz, n, conf_level=conf_level)[0]
        U2 = rz_ci(xz, n, conf_level=conf_level)[1]
        rho_r12_r13 = rho_rxy_rxz(xy, xz, yz)
        lower = xy - xz - pow((pow((xy - L1), 2) + pow((U2 - xz), 2) - 2 * rho_r12_r13 * (xy - L1) * (U2 - xz)), 0.5)
        upper = xy - xz + pow((pow((U1 - xy), 2) + pow((xz - L2), 2) - 2 * rho_r12_r13 * (U1 - xy) * (xz - L2)), 0.5)
        return lower, upper
    else:
        raise Exception('Wrong method!')

def independent_corr(xy, ab, n, n2 = None, twotailed=True, conf_level=0.95, method='fisher'):
    """
    Calculates the statistic significance between two independent correlation coefficients
    @param xy: correlation coefficient between x and y
    @param xz: correlation coefficient between a and b
    @param n: number of elements in xy
    @param n2: number of elements in ab (if distinct from n)
    @param twotailed: whether to calculate a one or two tailed test, only works for 'fisher' method
    @param conf_level: confidence level, only works for 'zou' method
    @param method: defines the method uses, 'fisher' or 'zou'
    @return: z and p-val
    """

    if method == 'fisher':
        xy_z = 0.5 * np.log((1 + xy)/(1 - xy))
        ab_z = 0.5 * np.log((1 + ab)/(1 - ab))
        if n2 is None:
            n2 = n

        se_diff_r = np.sqrt(1/(n - 3) + 1/(n2 - 3))
        diff = xy_z - ab_z
        z = abs(diff / se_diff_r)
        p = (1 - norm.cdf(z))
        if twotailed:
            p *= 2

        return z, p
    elif method == 'zou':
        L1 = rz_ci(xy, n, conf_level=conf_level)[0]
        U1 = rz_ci(xy, n, conf_level=conf_level)[1]
        L2 = rz_ci(ab, n2, conf_level=conf_level)[0]
        U2 = rz_ci(ab, n2, conf_level=conf_level)[1]
        lower = xy - ab - pow((pow((xy - L1), 2) + pow((U2 - ab), 2)), 0.5)
        upper = xy - ab + pow((pow((U1 - xy), 2) + pow((ab - L2), 2)), 0.5)
        return lower, upper
    else:
        raise Exception('Wrong method!')


def pulsealign(beh_ts, neural_ts, window=30, thresh=0.99):
    """
    Step through recorded pulses in chunks, correlate, and find matched pulses. Step 1 of a 2-step alignment process. 
    Step 2 uses these matched pulses for the regression and offset determination!
    
    """
    neural_blockstart = np.linspace(0, len(neural_ts)-window, window)
    beh_ipi = np.diff(beh_ts)
    neural_ipi = np.diff(neural_ts)

    print(f'{len(neural_blockstart)} blocks')
    blockR = [] 
    blockBehMatch = [] 

    for block in neural_blockstart:
        print('.', end =" ")
        neural_ix = np.arange(window-1) + block
        neural_d = neural_ipi[neural_ix.astype(int)]
        r = np.zeros(len(beh_ipi) - len(neural_d))
        p = r.copy() 
        for i in np.arange(len(beh_ipi)-len(neural_d)):
            temp_beh_ix = np.arange(window-1) + i 
            r_temp = np.corrcoef(neural_d, beh_ipi[temp_beh_ix])[0,1]
            r[i] = r_temp
        blockR.append(np.max(r))
        blockBehMatch.append(np.argmax(r))
    neural_offset = [] 
    good_beh_ms = [] 
    blockR = np.array(blockR)
    goodblocks = np.where(blockR>thresh)[0]
    for b in goodblocks:
        neural_ix = np.arange(window-1) + neural_blockstart[b]
        neural_offset.extend(neural_ts[neural_ix.astype(int)])
        beh_ix = np.arange(window-1) + blockBehMatch[b]
        good_beh_ms.extend(beh_ts[beh_ix])

    print(f'found matches for {len(neural_offset)} of {len(neural_ts)} pulses')

    return good_beh_ms, neural_offset


def sync_matched_pulses(beh_pulse, neural_pulse):
    """
    
    """
    bfix = beh_pulse[0]
    res = linregress(beh_pulse-bfix, neural_pulse)
    slope = res[0]
    offset = res[1]
    offset = offset - bfix*slope
    rval = res[2]

    return slope, offset, rval



