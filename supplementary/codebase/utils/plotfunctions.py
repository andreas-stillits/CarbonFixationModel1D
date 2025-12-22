"""

Module for plotting results

"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from codebase.utils.constants import TemporalExploration


def set_standard_layout() -> None:
    mpl.rcParams["mathtext.fontset"] = "stix"  # or 'dejavusans', 'cm', 'custom'
    mpl.rcParams["font.family"] = "STIXGeneral"  # Matches STIX math font
    # set tick font size
    mpl.rcParams["xtick.labelsize"] = 12
    mpl.rcParams["ytick.labelsize"] = 12
    # set default fontsize
    mpl.rcParams["font.size"] = 14


def get_blue_cmap(bounds: tuple[float, float, float]):
    min, max, res = bounds
    colors = [
        "#001261",
        "#02236C",
        "#023376",
        "#034481",
        "#06568C",
        "#156798",
        "#307DA6",
        "#4E92B4",
        "#71A8C4",
        "#94BED2",
        "#B3D1DF",
        "#D5E3E9",
    ]  # scicolor.get_cmap('oslo25').colors
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "Custom cmap", colors, N=21
    ).reversed()
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # force the first color entry to be
    cmaplist[0] = "white"
    # create the new map
    cmap = mpl.colors.LinearSegmentedColormap.from_list("Custom cmap", cmaplist, cmap.N)

    # define the bins and normalize
    bounds = np.linspace(min, max, res)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm


def plot_sensitivity_map(
    filename: str, bounds=(0, 26, 14), ax: plt.Axes | None = None
) -> mpl.collections.PolyQuadMesh:
    """Plot the sensitivity map from file"""
    set_standard_layout()
    sensitivity = 100 * np.loadtxt(filename, delimiter=";").T
    N = len(sensitivity)

    taus = np.exp(np.linspace(np.log(0.01), np.log(100), N))
    gammas = np.exp(np.linspace(np.log(0.01), np.log(100), N))

    # FULL PLOT

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    cmap, norm = get_blue_cmap(bounds)
    im = ax.pcolor(taus, gammas, sensitivity, shading="nearest", cmap=cmap, norm=norm)
    # cbar = plt.colorbar(im)
    # cbar.set_label(r'Sensitivity $\eta(\tau, \gamma)$ [%]')

    # lines = (0.248, 0.352, 0.570)  # 1%, 2%, 5% relative error lines
    # for line in lines:
    #     plt.vlines(line, 0.01, 100, color='grey', linestyle='--', linewidth=2)
    #
    # plt.text(0.012, 5, 'C: non-spatial', fontsize=14, color='black')
    # plt.text(2, 0.04, 'B: Spatially \n homogeneous', fontsize=14, color='black')
    # plt.text(2, 20, 'A: Spatially \n heterogeneous', fontsize=14, color='black')
    #
    if ax is None:
        ax.set_xlabel(
            r"Absorption balance $\tau = \sqrt{\langle K \rangle L^2 \; / \langle D\rangle}$"
        )
        ax.set_ylabel(r"Transport balance $\gamma = g_s L \; / \langle D \rangle$")
    ax.hlines(1, 0.01, 100, color="grey", linestyle="-.")
    ax.vlines(1, 0.01, 100, color="grey", linestyle="-.")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.01, 100)
    ax.set_ylim(0.01, 100)
    ax.set_aspect("equal")
    if ax is None:
        plt.show()
    return im


def plot_temporal_scan(
    quantity: str,
    case: str,
    rhomax: float = 0.5,
    ax: plt.Axes | None = None,
    vmax: float = 0.10,
    colorbar: bool = False,
) -> float:
    """Plot the temporal scan results for given quantity and case"""
    c = TemporalExploration()
    filename = (
        f"files/temporal_scanning/rhomax_{rhomax:.1f}_/{quantity}/variations_{case}.txt"
    )
    data = np.loadtxt(filename, delimiter=";")
    n_period, n_amp = data.shape
    periods = np.exp(np.linspace(np.log(c.period_min), np.log(c.period_max), n_period))
    amplitudes = np.linspace(c.amp_min, c.amp_max, n_amp)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    cmap, norm = get_blue_cmap((0, 0.26, 16))
    heatmap = ax.pcolor(
        periods, amplitudes, data.T, shading="nearest", cmap=cmap, norm=norm
    )
    if colorbar:
        plt.colorbar(heatmap, ax=ax)
    ax.set_xscale("log")
    ax.set_yscale("log")
    if ax is None:
        ax.set_xlabel("Period []")
        ax.set_ylabel("Amplitude [%]")
    ax.set_xlim(periods.min(), periods.max())
    ax.set_ylim(amplitudes.min(), amplitudes.max())
    if ax is None:
        plt.show()
    return np.max(data)
