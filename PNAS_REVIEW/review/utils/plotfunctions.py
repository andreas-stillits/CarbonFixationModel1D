"""

Module for plotting results

"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pyvista as pv
from dolfinx.plot import vtk_mesh
import adios4dolfinx as a4x
from mpi4py import MPI
from dolfinx import fem
import gmsh
from pathlib import Path
from review.utils.constants import TemporalExploration, Cases
from review.steady.solver import SteadySolver
from review.utils.profiles import StepProfile


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
    data = np.loadtxt(filename, delimiter=c.delimiter)
    periods, amplitudes = c.get_period_range(), c.get_amplitude_range()
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


def plot_all_rhomax_scans(quantity: str, case: str, vmax: float = 0.10) -> None:
    rhomax_values = [0.5, 0.4, 0.3, 0.2]
    fig, axs = plt.subplots(1, len(rhomax_values), figsize=(20, 5))
    axs = axs.flatten()
    for ax, rhomax in zip(axs, rhomax_values):
        maximum = plot_temporal_scan(quantity, case, rhomax=rhomax, ax=ax, vmax=vmax)
        ax.set_title(
            f"contrast  1 : {1/rhomax:.1f}  |  max = {maximum:.2f}", fontsize=12
        )

    plt.suptitle(f"Temporal Scan of {quantity} for case {case}", fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_timeseries(
    quantity: str,
    case: str,
    amplitude: float,
    period: float,
    rhomax: float = 0.5,
    ax: plt.Axes | None = None,
) -> None:
    constants = TemporalExploration()
    base_path = Path(f"files/temporal_scanning/rhomax_{rhomax:.1f}_/{quantity}/{case}/")
    timeseries_hom = np.loadtxt(
        base_path / f"timeseries_amp{amplitude:.2f}_per{period:.2f}_hom.txt",
        delimiter=constants.delimiter,
    )
    timeseries_het = np.loadtxt(
        base_path / f"timeseries_amp{amplitude:.2f}_per{period:.2f}_het.txt",
        delimiter=constants.delimiter,
    )

    time = np.linspace(0, constants.periods_to_run * period, len(timeseries_hom))

    variation = np.abs((timeseries_hom - timeseries_het) / (timeseries_het))

    if ax is None:
        fig, ax_ = plt.subplots(figsize=(10, 6))
    else:
        ax_ = ax
    ax_.plot(time, timeseries_hom, label="Homogeneous", color="blue")
    ax_.plot(time, timeseries_het, label="Heterogeneous", color="orange")

    ax__ = ax_.twinx()
    ax__.plot(time, variation, label="Relative Variation", color="k", linestyle=":")
    ax__.set_ylim(0, 0.25)
    ax__.set_ylabel("Relative Variation")
    print(f"Mean relative variation: {np.mean(variation):.4f}")
    print(f"Max relative variation: {np.max(variation):.4f}")

    ax_.set_title(
        f"Timeseries for {quantity} - Case {case}\nAmplitude: {amplitude}, Period: {period}"
    )
    ax_.set_xlabel("Time")
    ax_.set_ylabel("Assimilation rate A_n")
    ax_.set_ylim(0, np.max([timeseries_hom.max(), timeseries_het.max()]) * 1.1)
    ax_.grid()
    if ax is None:
        ax_.legend()
        plt.show()


def plot_steady_variation(
    case: str, rho: tuple[float, float, float] | None = None
) -> None:
    constants = TemporalExploration()
    if rho is None:
        rho = constants.rho
    delta = StepProfile(direction="down")
    delta.populate_rho(rho[0], rho[2])
    kappa = StepProfile(direction="up")
    kappa.populate_rho(rho[1], rho[2])
    params = Cases().get_case_params(case)
    solver_hom = SteadySolver(params)
    solver_het = SteadySolver(params, delta=delta.steadify(), kappa=kappa.steadify())
    domain_hom, sol_hom = solver_hom.solve()
    domain_het, sol_het = solver_het.solve()
    del solver_hom, solver_het
    del delta, kappa
    plt.plot(domain_hom, sol_hom, label="Homogeneous")
    plt.plot(domain_het, sol_het, label="Heterogeneous")
    plt.title(f"Steady State Solutions - Case {case}")
    plt.xlabel("x")
    plt.ylabel("Solution")
    plt.legend()
    plt.grid()
    plt.ylim(0, 1.1)
    plt.show()
    alpha_hom = 1 - sol_hom[0]
    alpha_het = 1 - sol_het[0]
    variation = np.abs((alpha_hom - alpha_het) / alpha_het)
    print(f"Case {case} - Variation: {variation:.4f}")


def plot_3d_solution(filename: str) -> None:
    """Plot the 3D solution from file"""
    mesh = a4x.read_mesh(filename, MPI.COMM_SELF)
    V = fem.functionspace(mesh, ("Lagrange", 1))
    uh = fem.Function(V)
    a4x.read_function(filename, uh, name="solution")
    #
    topology, cell_types, geometry = vtk_mesh(mesh, mesh.topology.dim)
    grid = pv.UnstructuredGrid(topology, cell_types, geometry)
    grid.point_data["uh"] = uh.x.array.real
    xmin, xmax, ymin, ymax, zmin, zmax = grid.bounds
    slices = grid.slice_orthogonal(
        x=(xmin + xmax) / 2, y=(ymin + ymax) / 2, z=(zmin + zmax) / 2
    )
    p = pv.Plotter(notebook=True)
    p.add_mesh(slices, scalars="uh", cmap="viridis", clim=[0.0, 1.05])
    p.add_mesh(grid.outline(), color="k")
    p.show_axes()
    p.show()


def show_mesh(filename: str) -> None:
    path = Path(filename)
    if path.suffix == ".msh" and path.is_file():
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("mesh from file")
        gmsh.merge(filename)
        gmsh.fltk.run()
        gmsh.finalize()
    else:
        raise FileNotFoundError(f"Mesh file {filename} not found.")
