""" 

Homogeneous profile function

"""


import numpy as np 

def homogeneous_solution(x: np.ndarray, params: tuple[float, float, float]) -> np.ndarray:
    """ Return a homogeneous profile function given parameters """
    tau, gamma, chi_ = params
    return chi_ + (1-chi_)/(1 + tau/gamma * np.tanh(tau)) * np.cosh(tau*(1-x))/np.cosh(tau)