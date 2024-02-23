import numpy as np
import pandas as pd


def get_linear_regression_results(X: np.array, y: np.array, constant=True, norm=True):
    if norm:
        X = normalization(X)
        y = normalization(y)
    if constant:
        X = np.vstack([X, np.ones(len(X))]).T
    betas = np.linalg.lstsq(X, y, rcond=None)[0]
    residuals = y - X.dot(betas)
    return betas, residuals


def normalization(arr):
    return (arr-arr.mean())/arr.std()
