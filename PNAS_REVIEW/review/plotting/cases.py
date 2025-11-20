""" 

Module to generate a heatmap matrix for a given case

"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from review.utils.constants import TemporalConstants
from review.utils.paths import get_case_path, get_temporal_scanning_path
from review.plotting.constants import PlottingConstants

pc = PlottingConstants()
tc = TemporalConstants()

VMIN = 0
VMAX = 0.10

def fetch_data(case: str, quantity: str, het: bool = False) -> np.ndarray:
    data_path = get_temporal_scanning_path(case, quantity)
    mean_alpha = np.loadtxt(data_path / "mean_alpha.txt", delimiter=";")
    if quantity == "K" and het:
        alpha_reduced = np.loadtxt(data_path / "alpha_het.txt", delimiter=";")
    else:
        alpha_reduced = np.loadtxt(data_path / "alpha_hom.txt", delimiter=";")
    heat = (alpha_reduced - mean_alpha) / mean_alpha
    return heat


def generate_casemap(case: str) -> None:
    case_path = get_case_path(case)
    fig, axs = plt.subplots(2, 2, figsize=(10,10), sharex=True, sharey=True)
    amplitudes = np.exp(np.linspace(np.log(tc.amp_min), np.log(tc.amp_max), tc.n_amp))
    periods = np.exp(np.linspace(np.log(tc.period_min), np.log(tc.period_max), tc.n_period))
    # Ca 
    data = fetch_data(case, "Ca")
    generate_heatmap(axs[1,0], data, amplitudes, periods)
    axs[1,0].set_title("Ca", fontsize=16)
    # gs
    data = fetch_data(case, "gs")
    generate_heatmap(axs[1,1], data, amplitudes, periods)
    axs[1,1].set_title("gs", fontsize=16)
    # K homogeneous
    data = fetch_data(case, "K", het=False)
    generate_heatmap(axs[0,0], data, amplitudes, periods)
    axs[0,0].set_title("K homogeneous", fontsize=16)
    # K heterogeneous
    data = fetch_data(case, "K", het=True)
    generate_heatmap(axs[0,1], data, amplitudes, periods)
    axs[0,1].set_title("K heterogeneous", fontsize=16)
    plt.suptitle(f"Case {case}", fontsize=20)
    plt.show()


def generate_heatmap(ax: plt.Axes, data: np.ndarray, amplitudes: np.ndarray, periods: np.ndarray) -> None:
    heatmap = ax.pcolor(periods, amplitudes, data, shading='auto', cmap=pc.CMAP, vmin=VMIN, vmax=VMAX) 
    plt.colorbar(heatmap, ax=ax)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Period []")
    ax.set_ylabel("Amplitude []")
    ax.set_xlim(tc.period_min, tc.period_max)
    ax.set_ylim(tc.amp_min, tc.amp_max)
