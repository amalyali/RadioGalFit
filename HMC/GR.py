#  Gelman-Rubin test for MCMC chains
#  Author: Michelle Knights (2013)

from __future__ import print_function, division
import numpy as np


def gelman(chains, burn=0):
    # Chains are assumed to be the same length

    M = len(chains)
    p = len(chains[0][0,:])
    n = len(chains[0])

    means = np.zeros([M,p]) # Contains the mean for each chain
    variances = np.zeros([M,p])  # Contains the variance for each chain

    for m in range(M):
        means[m,:] = np.mean(chains[m][burn:,:], axis=0)
        variances[m,:] = np.std(chains[m][burn:,:], axis=0) ** 2

    chain_mean = np.mean(means, axis=0)  # Get the between-chain mean

    B = n / (M - 1) * np.sum((means - chain_mean) ** 2, axis=0)  # Between-chain variance
    W = 1.0 / M * np.sum(variances, axis=0)  # Within chain variance

    V = (float)(n - 1) / n * W + (float)(M + 1) / n / M * B  # Posterior marginal variance

    return np.sqrt(V / W)

def convergence_test(chains, jump=500, tol=0.1): # 0.01):

    # Prune chains to same length
    n = len(chains[0])
    for m in range(1, len(chains)):
        if len(chains[m])<n:
            n = len(chains[m])
    for m in range(len(chains)):
        chains[m] = chains[m][:n,:]

    results = []
    steps = []
    burn = 0
    for step in range(burn, n, jump):
        print('Step', step,'of', n)

        gr = gelman(chains, burn=step)
        steps.append(step)

        if len(results) == 0:
            results = gr
        else:
            results = np.vstack((results, gr))

    converged = np.any(results <= 1+tol, axis=0)
    print(np.sum(converged==False), 'parameters did not converge')
    
    return steps, results
