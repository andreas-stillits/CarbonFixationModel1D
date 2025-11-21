""" 

Module for plotting results

"""

import numpy as np 
import matplotlib.pyplot as plt
from review.plotting.constants import PlottingConstants 

pc = PlottingConstants()

def plot_sensitivity_map(filename: str, bounds=(0, 26, 14)):
    """ Plot the sensitivity map from file """
    pc.set_standard_layout()
    sensitivity = 100*np.loadtxt(filename, delimiter=';').T
    N = len(sensitivity)

    taus   = np.exp(np.linspace(np.log(0.01), np.log(100), N))
    gammas = np.exp(np.linspace(np.log(0.01), np.log(100), N))

    # FULL PLOT
    fig = plt.figure(figsize=(8,6))
    cmap, norm = pc.get_blue_cmap(bounds)
    im = plt.pcolor(taus, gammas, sensitivity, shading='nearest', cmap=cmap, norm=norm)
    cbar = plt.colorbar(im)

    lines = pc.tau_lines_sensitivity
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
    