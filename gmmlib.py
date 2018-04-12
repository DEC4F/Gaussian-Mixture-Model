import csv, copy, gzip, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.stats import chi2


def dataset_factory(mu, sig, size):
    """
    creats a dataset as numpy array based on the input mean, covariance matrix and total number of datapoints
    """
    assert len(mu) == len(sig)
    size_c = int(size/len(mu)) # create same num of datapoints for each cluster
    X = []
    Y = []
    for c in range(len(mu)):
        x_c, y_c = np.random.multivariate_normal(mu[c], sig[c], size_c).T
        X = np.concatenate((X, x_c))
        Y = np.concatenate((Y, y_c))
    data = np.array([X, Y]).T
    return data


def expectation(data, means, covs, weights):
    """
    returns the expectation of each datapoint wrt each cluster
    """
    T = len(data)  # number of datapoint
    K = len(means)  # number of clusters
    p_nk = np.zeros((T, K))  # a matrix that stores the pdf of each datapoint wrt each cluster
    
    for i in range(T):
        for c in range(K):
            p_nk[i, c] = weights[c] * multivariate_normal.pdf(data[i], means[c], covs[c])
    p_nk = p_nk / p_nk.sum(axis=1)[:, np.newaxis] # normalize the pdf by dividing the row sum
   
    return p_nk


def log_likelihood(data, means, covs, weights):
    """
    calculate current log likelihood of all datapoints
    """
    K = len(means)
    F = len(data[0])
    ll = 0
    
    for d in data:
        Z = np.zeros(K)

        for k in range(K):
            delta = np.array(d) - means[k]
            exp = np.dot(delta.T, np.dot(np.linalg.inv(covs[k]), delta))
            # log likelihood of this datapoint affected by this cluster
            Z[k] += np.log(weights[k])
            Z[k] -= 1/2 * (F * np.log(2 * np.pi) + np.log(np.linalg.det(covs[k])) + exp)
        # sum of log likelihood of all datapoints
        ll += np.max(Z) + np.log(np.sum(np.exp(Z - np.max(Z))))
            
    return ll


def next_mean (data, p_nk, count):
    """
    update mu value for M-step
    """
    T = len(data) # number of datapoints
    K = len(np.sum(p_nk, axis=0)) # number of clusters
    new_means = [np.zeros(len(data[0]))]*K # new mu value for each cluster
    for c in range(K):
        cumsum = 0
        for i in range(T):
            cumsum += p_nk[i, c] * data[i]
        new_means[c] = cumsum * 1 / count[c] # weighted by marginal prob

    return new_means


def next_cov (data, p_nk, mu, count):
    """
    update Sigma value (covariance matrix) for M-step
    """
    T = len(data) # number of datapoints
    K = len(count) # number of clusters
    F = len(data[0]) # number of features
    new_cov = [np.zeros((F, F))] * K # new covariance matrix for each cluster

    for c in range(K):
        cumsum = np.zeros((F, F))

        for i in range(T):
            cumsum += p_nk[i, c] * np.outer((data[i] - mu[c]), (data[i] - mu[c]).T)
        new_cov[c] = cumsum/count[c] # weighted by marginal prob

    return new_cov


def next_weights (p_nk, count):
    """
    update lambda value for M-step
    """
    K = len(count) # number of clusters
    weights = np.zeros(K)

    for c in range(K):
        weights[c] = count[c] / np.sum(count) # calculate new weight of each cluster
    
    return weights


def maximization(data, means, covs, weights, p_nk):
    """
    updates mu, Sigma and lambda value
    """
    count = np.sum(p_nk, axis=0)
    
    new_means = next_mean(data, p_nk, count)
    new_covs = next_cov(data, p_nk, new_means, count)
    new_weights = next_weights(p_nk, count)

    return new_means, new_covs, new_weights


def EM_algo(data, init_means, init_covs, init_weights, iterations=100, thresh=0.001):
    """
    perform expectation maximization algorithm on the input dataset
    return the Gaussian Mixture Model when max iteration is reached or change in log-likelihood is lower than threshold
    """
    means = init_means[:]
    covs = init_covs[:]
    weights = init_weights[:]
    log_like = log_likelihood(data, means, covs, weights)
    ll_trace = [log_like]
    curr_iters = 0

    while iterations > curr_iters:
        curr_iters += 1
        # E-step: update label on each datapoint
        p_nk = expectation(data, means, covs, weights)
        # M-step: update parameters of each cluster based on actual data
        means, covs, weights = maximization(data, means, covs, weights, p_nk)
        # update log likelihood
        log_like = log_likelihood(data, means, covs, weights)
        # append to trace array
        ll_trace.append(log_like)
        if (log_like - ll_trace[-2] < thresh) and (log_like > -np.inf):
            break
    
    final_gmm = []
    for c in range(len(means)):
        final_gmm.append({'mean' : means[c], 'covariance' : covs[c], 'weight' : weights[c]})

    return final_gmm, ll_trace


def plotGaussianModel2D(mu, sigma, pltopt='k'):
    if sigma.any():
        # calculate ellipse constants
        c = chi2.ppf(0.9, 2) # use confidence interval 0.9
        # get eigen vector and eigen values
        eigenValue, eigenVector = np.linalg.eig(sigma)
        # calculate points on ellipse
        t = np.linspace(0, 2*np.pi, 100) # draw 100 points
        u = [np.cos(t), np.sin(t)]
        w = c * eigenVector.dot(np.diag(np.sqrt(eigenValue)).dot(u))
        z = w.T + mu
    else:
        z = mu
    # plot ellipse by connecting sample points on curve
    plt.plot(z[:,0], z[:,1], pltopt)
    plt.xlabel('PCA 1'); plt.ylabel('PCA2')
    plt.title('Clustering of 2D GMM')
    

def colorPicker(index):
    colors = 'cmykrgb'
    return colors[np.remainder(index, len(colors))]


def gmmplot(data, gmm):
    # plot data points
    plt.scatter(data[:, 0], data[:, 1], s=4)
    # plot Gaussian model
    for index, model in enumerate(gmm):
        plotGaussianModel2D(model['mean'], model['covariance'], colorPicker(index))


def traceplot(ll_trace):
    plt.plot(ll_trace, 'o-')
    plt.xlabel('Iterations'); plt.ylabel('Log-likelihood')
    plt.title('Change of Log-likelihood')
    

def doubleplot(data, gmm, ll_trace, size):
    """
    plot the traceplot and gmmplot
    """
    fig = plt.figure(figsize=(size[0], size[1])) 
    plt.subplot(1, 2, 1)
    gmmplot(data, gmm)
    plt.subplot(1, 2, 2)
    traceplot(ll_trace)


def random_init(data, K):
    """
    return randomly initialized mu, Sigma and lambda value with K as number of anticipated clusters
    """
    selected = np.random.choice(len(data), K, replace = False)
    init_mu = [data[i] for i in selected]
    init_sig = [np.cov(data, rowvar=0)] * K
    init_lambda = [1/K]*K

    return init_mu, init_sig, init_lambda


def doPCA(dataset, n):
    """
    perform principal component analysis on a dataset of high dimensionality
    returns the reduced dataset, eigen vectors and eigen values
    """
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n)
    transformed_data = pca.fit(dataset).transform(dataset)
    eigen_vec = pca.components_
    eigen_val = pca.explained_variance_
    
    return transformed_data, eigen_vec, eigen_val