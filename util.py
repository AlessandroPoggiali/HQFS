import math
import numpy as np
import pandas as pd
from numpy.linalg import norm

def rbo(S,T, p= 0.9):
    """ Takes two lists S and T of any lengths and gives out the RBO Score
    Parameters
    ----------
    S, T : Lists (str, integers)
    p : Weight parameter, giving the influence of the first d
        elements on the final score. p<0<1. Default 0.9 give the top 10 
        elements 86% of the contribution in the final score.
    
    Returns
    -------
    Float of RBO score
    """
    
    # Fixed Terms
    k = max(len(S), len(T))
    x_k = len(set(S).intersection(set(T)))
    
    summation_term = 0

    # Loop for summation
    # k+1 for the loop to reach the last element (at k) in the bigger list    
    for d in range (1, k+1): 
            # Create sets from the lists
            set1 = set(S[:d]) if d < len(S) else set(S)
            set2 = set(T[:d]) if d < len(T) else set(T)
            
            # Intersection at depth d
            x_d = len(set1.intersection(set2))

            # Agreement at depth d
            a_d = x_d/d   
            
            # Summation
            summation_term = summation_term + math.pow(p, d) * a_d

    # Rank Biased Overlap - extrapolated
    rbo_ext = (x_k/k) * math.pow(p, k) + ((1-p)/p * summation_term)

    return rbo_ext

def compute_angle(x, y):
    x = np.array(x)
    y = np.array(y)
    return np.arccos((x @ y) / (norm(x) * norm(y)))  # in range [0, pi]

def generate_dataset(n_samples, n_features, n_informative, seed=123, mu=0, std=0.05):    
    
    df = pd.DataFrame()
    #n_redundant = n_features - n_informative

    for f in range(n_informative):
        np.random.seed(seed*f)
        feature = "f"+str(f)
        df[feature] = np.random.uniform(low=-1.0, high=1.0, size=n_samples)

    for f in range(n_informative, n_features):
        np.random.seed(seed*f)
        feature = "f"+str(f)
        df[feature] = np.random.normal(loc=mu, scale=std, size=n_samples)

    return df