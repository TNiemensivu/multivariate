import numpy as np

def sample_mean (data):
    n = len(data)
    ones = np.ones((n,1))
    return np.divide(np.matmul(np.transpose(data),ones), n)

def sample_covariance (data):
    n = len(data)
    I = np.identity(n)
    ones = np.ones((n,1))
    H = I-np.divide(np.matmul(ones, np.transpose(ones)),n)
    return np.divide(np.matmul(np.transpose(data), np.matmul(H, data)),n-1)


def sample_correlation (data):
    S = sample_covariance(data)
    p = len(S)
    V = np.divide(np.identity(p), np.sqrt(np.diag(S)))
    return np.matmul(V, np.matmul(S,V))

def standardize_data (data):
    S = sample_covariance(data)
    n = len(data)
    I = np.identity(n)
    ones = np.ones((n,1))
    H = I-np.divide(np.matmul(ones, np.transpose(ones)),n)
    sqrt_s  = None
    return np.matmul(H, np.matmul(data, sqrt_s))

data = [[2,1],[1,2],[1,4]]
print(sample_mean(data))
print(sample_covariance(data))
print(sample_correlation(data))
print(standardize_data(data))