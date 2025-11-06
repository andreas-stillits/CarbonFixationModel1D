"""  

Module for plotting steady-state solutions and figure 3C.

"""

import numpy as np
import matplotlib.pyplot as plt
from utils import extract_solution_from_file
import sys 
sys.path.append('../../../modules')
import matplotlib_config as mconf



def plot_simple_solution(filename: str):
    """ Plot the solution from the steady solver """
    domain, solution = extract_solution_from_file(filename)
    plt.plot(domain, solution)
    plt.xlabel("z")
    plt.ylabel("CO2 concentration")
    plt.title("Steady-state CO2 profile")
    plt.xlim(0, 1)
    plt.ylim(0, 1.05)
    plt.grid()
    plt.show()
    return domain, solution

def compare_two_solutions(domain1: np.ndarray, solution1: np.ndarray, domain2: np.ndarray, solution2: np.ndarray):
    """ Compare two solutions by plotting them together """
    plt.plot(domain1, solution1, label="Solution 1")
    plt.plot(domain2, solution2, label="Solution 2", linestyle='--')
    plt.xlabel("z")
    plt.ylabel("CO2 concentration")
    plt.title("Comparison of Steady-state CO2 profiles")
    plt.xlim(0, 1)
    plt.ylim(0, 1.05)
    plt.grid()
    plt.legend()
    plt.show()


def plot_sensitivity_map(filename: str):
    """ Plot the sensitivity map from file """
    colorI, colorII, colorIII = mconf.set_standard_layout()
    sensitivity = 100*np.loadtxt(filename, delimiter=';')
    N = len(sensitivity)

    taus   = np.exp(np.linspace(np.log(0.01), np.log(100), N))
    gammas = np.exp(np.linspace(np.log(0.01), np.log(100), N))
    bounds = (0, 26, 14) # color map boundaries, 0-26% sensitivity, 14 steps 

    # FULL PLOT
    fig = plt.figure(figsize=(8,6))
    cmap, norm = mconf.my_cmap(*bounds)
    im = plt.pcolor(taus, gammas, sensitivity, shading='nearest', cmap=cmap, norm=norm)
    cbar = plt.colorbar(im)

    lines = [0.248, 0.352, 0.570] #1%, 2%, 5% relative error
    for line in lines:
        plt.vlines(line, 0.01, 100, color='grey', linestyle='--', linewidth=2)
    #  
    plt.text(0.012, 5, 'C: non-spatial', fontsize=14, color='black')
    plt.text(2, 0.04, 'B: Spatially \n homogeneous', fontsize=14, color='black')
    plt.text(2, 20, 'A: Spatially \n heterogeneous', fontsize=14, color='black')
    #
    cbar.set_label(r'Sensitivity $\eta(\tau, \gamma)$ [%]')
    plt.xlabel(r'Absorption balance $\tau = \sqrt{\langle K \rangle L^2 \; / \langle D\rangle}$')
    plt.ylabel(r'Transport balance $\gamma = g_s L \; / \langle D \rangle$')
    plt.hlines(1, 0.01, 100, color='grey', linestyle='-.')
    plt.vlines(1, 0.01, 100, color='grey', linestyle='-.')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(0.01, 100)
    plt.ylim(0.01, 100)
    plt.gca().set_aspect('equal')
    plt.show()