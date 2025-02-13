from typing import Callable
import numpy as np


def optimal_weights(o, y, q=0):
    d = o.shape[1]
    return np.linalg.inv(q * np.identity(d) + o.T @ o) @ o.T @ y


# Univar
def design_matrix_univar(x, d):
    powers = np.arange(0, d)
    design_matrix = x[:, np.newaxis] ** powers
    return design_matrix


def predict_func_univar(w):
    return np.polynomial.polynomial.Polynomial(w)


def train_univar(x, y, d, q=0):
    o = design_matrix_univar(x, d)
    w = optimal_weights(o, y, q)
    return predict_func_univar(w)


# Bivariate
def design_matrix_bivar(x1, x2, d):
    terms = []
    for i in range(d):
        for j in range(d - i):
            terms.append((x1**i) * (x2**j))
    design_matrix = np.vstack(terms).T
    return design_matrix


def predict_func_bivar(w, d):
    def predict(x1: float, x2: float) -> float:
        total = 0
        idx = 0
        for i in range(d):
            for j in range(d - i):
                total += w[idx] * (x1**i) * (x2**j)
                idx += 1
        return total

    return predict


def train_bivar(x1, x2, y, d, q=0):
    o = design_matrix_bivar(x1, x2, d)
    w = optimal_weights(o, y, q)
    return predict_func_bivar(w, d)


# For N-variant: Change binomial to n-nomial (from bivar) and predict func
# Else: keep same
