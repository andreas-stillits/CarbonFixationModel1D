"""  

Module for during parameter scan to reproduce figure 3C

"""

import numpy as np
from tqdm import tqdm 
from ss_solver import solver  
from dolfinx import mesh
from mpi4py import MPI
from utils import get_homogeneous_chii, exp_down, exp_up, extract_solution_from_objects
import functools


def search_exponential(beta_range: tuple[float, float] = (0.1, 3.0), domain: mesh.Mesh | None = None, chi_: float = 0.1, param_num: int = 10, comb_num: int = 5, domain_resolution: int = 100):
    """ 
    Perform a parameter scan over tau and gamma with exponential delta(z) and kappa(z)
    
    Parameters
    ----------
    beta_range : tuple[float, float]
        The min and max beta values for the exponential functions
    domain : mesh.Mesh | None
        The mesh domain to solve on. If None, a default mesh will be used.
    chi_ : float
        The chi_ parameter
    param_num : int
        Number of tau and gamma parameters to sample
    comb_num : int
        Number of combinations of beta values to sample for each (tau, gamma)
    
    Returns
    -------
    taus : np.ndarray
        Array of tau values sampled
    gammas : np.ndarray
        Array of gamma values sampled
    sensitivities : np.ndarray
        2D array of sensitivities for each (tau, gamma) pair
    """
    param_range = np.exp(np.linspace(np.log(0.01), np.log(100), param_num))
    betas = np.linspace(beta_range[0], beta_range[1], comb_num)
    sensitivities = np.zeros((param_num, param_num))
    if domain is None:
        domain = mesh.create_interval(MPI.COMM_WORLD, domain_resolution, [0.0, 1.0])
    for i in tqdm(range(param_num)):
        for j in range(param_num): 
            tau = param_range[i]
            gamma = param_range[j] 
            params = [tau, gamma, chi_]
            chii_ref = get_homogeneous_chii(params)
            samples = comb_num**2
            relative_drawdowns = np.zeros(samples)
            index = 0
            for k in range(comb_num):
                for l in range(comb_num):
                    delta = functools.partial(exp_down, beta=betas[k])
                    kappa = functools.partial(exp_up, beta=betas[l])
                    domain, solution = solver(params, delta=delta, kappa=kappa, domain=domain)
                    domain_array, solution_array = extract_solution_from_objects(domain, solution)
                    chii = solution_array[0]
                    relative_drawdowns[index] = (chii_ref - chii)/(1 - chii_ref)
                    index += 1
            sensitivities[i, j] = np.sqrt(np.sum(relative_drawdowns**2)/samples)
    return param_range, param_range, sensitivities.T
