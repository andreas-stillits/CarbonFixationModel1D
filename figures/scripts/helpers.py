import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def std_layout(ax: plt.Axes, vmin: float = 0.01, vmax: float = 100) -> None:
    """Helper function to set standard plot settings for (tau,gamma) figures."""
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.set_xlabel(r"Absorption balance $\tau$ []")
    ax.set_ylabel(r"Transport balance $\gamma$ []")
    ax.plot([1, 1], [vmin, vmax], color="grey", linestyle="-.", zorder=2)
    ax.plot([vmin, vmax], [1, 1], color="grey", linestyle="-.", zorder=2)
