import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap, BoundaryNorm, to_rgb
from pathlib import Path

# set up plotting config
mpl.rcParams["mathtext.fontset"] = "stix"  # or 'dejavusans', 'cm', 'custom'
mpl.rcParams["font.family"] = "STIXGeneral"  # Matches STIX math font
# set tick font size
mpl.rcParams["xtick.labelsize"] = 12
mpl.rcParams["ytick.labelsize"] = 12
# set default fontsize
mpl.rcParams["font.size"] = 16


def map_data_values(
    df: pd.DataFrame, map: np.ndarray, vmin: float = 0.01, vmax: float = 100
) -> np.ndarray:
    """
    Function that maps (tau, gamma) of data points to values of a provided 2D map.
    Linear interpolation in log-space is used to determine values that fall between grid points,
    while out-of-bounds values are assigned the value of the nearest grid point to avoid extrapolation.

    Args:
        df (pd.DataFrame): pandas data frame containing columns "tau" and "gamma"
        map (np.ndarray): 2D map for interpolation, typically a relative error in An
        vmin (float): lower limit of tau, gamma space
        vmax (float): upper limit of tau, gamma space
    Returns:
        np.ndarray: a 1D array of the mapped values in row-order
    """
    taus_data = df["tau"].to_numpy()
    gammas_data = df["gamma"].to_numpy()
    N_gamma, N_tau = map.shape
    taus = np.exp(np.linspace(np.log(vmin), np.log(vmax), N_tau))
    gammas = np.exp(np.linspace(np.log(vmin), np.log(vmax), N_gamma))
    log_taus = np.log(taus)
    log_gammas = np.log(gammas)

    values = []
    for tau_data, gamma_data in zip(taus_data, gammas_data):
        # Handle out-of-bounds values by clipping to grid range via nearest neighbour
        if (
            (tau_data <= taus[0])
            or (tau_data >= taus[-1])
            or (gamma_data <= gammas[0])
            or (gamma_data >= gammas[-1])
        ):
            idx_tau = int(np.argmin(np.abs(taus - tau_data)))
            idx_gamma = int(np.argmin(np.abs(gammas - gamma_data)))
            values.append(map[idx_gamma, idx_tau])
            continue

        # Work in log-space for interpolation weights
        log_tau_data = np.log(tau_data)
        log_gamma_data = np.log(gamma_data)

        # Find indices of the two neighbouring grid points in each direction (in log-space)
        idx_tau_upper = int(np.searchsorted(log_taus, log_tau_data, side="right"))
        idx_tau_lower = idx_tau_upper - 1
        idx_gamma_upper = int(np.searchsorted(log_gammas, log_gamma_data, side="right"))
        idx_gamma_lower = idx_gamma_upper - 1

        x0, x1 = log_taus[idx_tau_lower], log_taus[idx_tau_upper]
        y0, y1 = log_gammas[idx_gamma_lower], log_gammas[idx_gamma_upper]

        f00 = map[idx_gamma_lower, idx_tau_lower]
        f10 = map[idx_gamma_lower, idx_tau_upper]
        f01 = map[idx_gamma_upper, idx_tau_lower]
        f11 = map[idx_gamma_upper, idx_tau_upper]

        tx = (log_tau_data - x0) / (x1 - x0)
        ty = (log_gamma_data - y0) / (y1 - y0)

        value = (
            (1 - tx) * (1 - ty) * f00
            + tx * (1 - ty) * f10
            + (1 - tx) * ty * f01
            + tx * ty * f11
        )
        values.append(value)

    return np.array(values)


def std_frame(ax: plt.Axes, df: pd.DataFrame) -> None:
    """
    Set standard plot settings for (tau,gamma) figures.

    Args:
        ax (plt.Axes): The axes object to set the frame settings for
        df (pd.DataFrame): The dataframe containing the data to plot

    Returns:
        None
    """
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.01, 100)
    ax.set_ylim(0.01, 100)
    ax.plot([1, 1], [0.01, 100], color="lightgrey", linestyle="-.", zorder=2)
    ax.plot([0.01, 100], [1, 1], color="lightgrey", linestyle="-.", zorder=2)
    ax.scatter(df["tau"], df["gamma"], color="black", marker="o", zorder=3, s=35)
    return


def plot_map(
    ax: plt.Axes,
    map: np.ndarray,
    cmap: ListedColormap,
    norm: BoundaryNorm,
    vmin: float = 0.01,
    vmax: float = 100,
):
    """
    Helper function for plotting a 2D map in (tau, gamma) space with the correct scaling and color mapping.
    Args:
        ax (plt.Axes): The axes object to plot on
        map (np.ndarray): 2D array of values to plot, typically a relative error in An
        cmap (ListedColormap): Colormap to use for plotting the map
        norm (BoundaryNorm): Normalization to use for mapping values to colors
        vmin (float): lower limit of tau, gamma space
        vmax (float): upper limit of tau, gamma space
    Returns:
        The result of ax.pcolormesh, which is a QuadMesh object
    """

    N = map.shape[0]
    taus = np.exp(np.linspace(np.log(vmin), np.log(vmax), N))
    gammas = np.exp(np.linspace(np.log(vmin), np.log(vmax), N))
    im = ax.pcolormesh(taus, gammas, map, shading="nearest", cmap=cmap, norm=norm)

    return im


def make_colormap(base_color: str, bounds: list) -> tuple[ListedColormap, BoundaryNorm]:
    """
    Create a colormap and normalization for plotting error maps.

    Args:
        base_color (str): named colormap recognized in matplotlib, e.g. "Reds", "Greens", "Blues"
        bounds (list): list of ticks to use for binning
    Returns:
        cmap (ListedColormap), norm (BoundaryNorm) for plottig
    """
    cmap = plt.get_cmap(base_color, len(bounds))
    colors = cmap(range(len(bounds) - 1))
    colors[0] = (1, 1, 1, 1)  # set the first color to white
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(bounds, cmap.N)
    return cmap, norm


def std_histogram(ax: plt.Axes, xlabel: str = "Error (%)") -> None:
    """
    Helper function for standard histogram formatting
    Args:
        ax (plt.Axes): The axes object to plot on
        xlabel (str): The x-axis label
    Returns:
        None

    """
    ax.set_ylabel("Count")
    ax.set_xlabel(xlabel)
    ax.set_yscale("log")
    return


def describe_errors(errors: np.ndarray, *thresholds: float) -> None:
    """
    Helper function to print descriptive statistics of error distributions, including mean, median, max, min, and percentage of errors below specified thresholds.
    Args:
        errors (np.ndarray): 1D array of error values to describe
        thresholds (float): variable number of threshold values to calculate percentage of errors below each threshold
    Returns:
        None
    """
    print(f"Mean error: {np.mean(errors):.2f}%")
    print(f"Median error: {np.median(errors):.2f}%")
    print(f"Max error: {np.max(errors):.2f}%")
    print(f"Min error: {np.min(errors):.2f}%")
    for threshold in thresholds:
        print(
            f"Percentage of errors below {threshold}%: {100 * np.sum(errors < threshold)/len(errors):.2f}%"
        )
    print("\n")
    return


def load_error_map(path: str) -> np.ndarray:
    """
    Tiny I/O helper function to load error maps from text files, with error handling for missing files and scaling to percentage values.
    Args:
        path (str): file path to the error map text file, expected to be a 2D array with values separated by semicolons
    Returns:
        np.ndarray: 2D array of error values loaded from the file, scaled to percentage values
    Raises:
        FileNotFoundError: if the specified file does not exist
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Error map file not found: {path}. Please use global variables as indicated to change plot settings"
        )

    return 100 * np.loadtxt(path, delimiter=";").T


def plot_1d_phase_space(df: pd.DataFrame) -> None:
    """
    Function for plotting data on top of the 1D phase space as in published figure 2D
    Args:
        df (pd.DataFrame): pandas data frame containing columns "tau" and "gamma"
    Returns:
        None
    """
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    colorI, colorII, colorIII = "#71a8c4", "#eacebd", "#f8a17b"
    N = 400
    vmin, vmax = 0.01, 100
    taus = np.exp(np.linspace(np.log(vmin), np.log(vmax), N))
    borders = lambda taus, sigma: sigma * taus * np.tanh(taus)
    # shading
    ax.fill_between(taus, borders(taus, 1), vmin, color=colorI)
    ax.fill_between(taus[taus <= 1], vmax, borders(taus[taus <= 1], 1), color=colorII)
    ax.fill_between(taus[taus >= 1], vmax, borders(taus[taus >= 1], 1), color=colorIII)
    # lines
    ax.plot(taus, borders(taus, 1), color="black", linestyle="-", linewidth=2)
    ax.plot(
        taus, borders(taus, np.sqrt(50)), color="black", linestyle="--", linewidth=1
    )
    ax.plot(
        taus, borders(taus, np.sqrt(10)), color="black", linestyle="--", linewidth=1
    )
    ax.plot(
        taus, borders(taus, 1 / np.sqrt(10)), color="black", linestyle="--", linewidth=1
    )
    ax.plot(
        taus, borders(taus, 1 / np.sqrt(50)), color="black", linestyle="--", linewidth=1
    )
    # data
    std_frame(ax, df)
    # annotations
    plt.text(1.3, 0.015, r"I: Stomata limited", color="black", fontsize=16)
    plt.text(
        0.012,
        2,
        r"II: Absorption capacity" + "\n     limited",
        color="black",
        fontsize=16,
    )
    plt.text(1.3, 60, r"III: IAS limited", color="black", fontsize=16)
    return


def plot_error_map(
    df: pd.DataFrame, error_map: np.ndarray, base_color="Reds"
) -> np.ndarray:
    """
    Function for plotting a 2D errror map in (tau, gamma) space with a histogram of the error distribution across data points.
    The error values for each data point are mapped from the 2D error map using linear interpolation in log-space,
    and descriptive statistics of the error distribution are printed to the console.

    Args:
        df (pd.DataFrame): pandas data frame containing columns "tau" and "gamma"
        error_map (np.ndarray): 2D array of error values in (tau, gamma) space, typically a relative error in An
        base_color (str): named colormap recognized in matplotlib, e.g. "Reds", "Greens", "Blues" to use for plotting the error map
    Returns:
        np.ndarray: 1D array of error values mapped to each data point, in the same order as the input dataframe
    """
    fig, (ax_hist, ax_map) = plt.subplots(
        2, 1, gridspec_kw={"height_ratios": [1, 4], "hspace": 0.25}, figsize=(7, 7)
    )
    cmap, norm = make_colormap(base_color, bounds=[0, 5, 10, 25, 50, 100, 200])

    # map error values to data points
    error_map_data = map_data_values(df, error_map)

    # create histogram of errors
    bins = np.linspace(0, 80, 41)
    ax_hist.hist(
        error_map_data, bins=bins, alpha=0.7, color="#3a3939ff", edgecolor="black"
    )
    std_histogram(ax_hist)

    # plot 3D error map and data
    im3d = plot_map(ax_map, error_map, cmap, norm)
    std_frame(ax_map, df)

    fig.colorbar(
        im3d, ax=[ax_map, ax_hist], label=r"Relative error in $A_N$ prediction (%)"
    )
    plt.show()

    return error_map_data


def transform_heatmap(map: np.ndarray, threshold: float = 5) -> np.ndarray:
    """
    Helper function to transform a 2D error map into a binary decision boundary map,
    where values below a specified threshold are set to 0 and values above the threshold are set to 1.
    Args:
        map (np.ndarray): 2D array of error values in (tau, gamma) space, typically a relative error in An
        threshold (float): threshold value to use for creating the binary decision boundary map
    Returns:
        np.ndarray: 2D binary array where values below the threshold are 0 and values above the threshold are 1
    """
    mask = map < threshold
    transformed_map = map.copy()
    transformed_map[mask] = 0
    transformed_map[~mask] = 1
    return transformed_map


def binary_colormap(hexcolor: str):
    """
    Helper function to create a binary colormap where 0 is transparent and 1 is the specified color
    Args:
        hexcolor (str): hex code for the color to use for values of 1 in the binary colormap, e.g. "#b21117"
    Returns:
        cmap (ListedColormap), norm (BoundaryNorm) for plotting the binary map
    """
    colors = [[1, 1, 1, 0], to_rgb(hexcolor) + (1,)]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm([0, 1], cmap.N)
    return cmap, norm


def plot_decision_boundaries(
    df: pd.DataFrame,
    error_ias_gradients: np.ndarray,
    error_lateral_gradients: np.ndarray,
    error_heterogeneity: np.ndarray,
    error_threshold: float,
) -> None:
    """
    Function for plotting decision boundaries based on error thresholds as in published figure 4
    Args:
        df (pd.DataFrame): pandas data frame containing columns "tau" and "gamma"
        error_ias_gradients (np.ndarray): 2D array of error values in (tau, gamma) space for omitting IAS CO2 gradients
        error_lateral_gradients (np.ndarray): 2D array of error values in (tau, gamma) space for omitting lateral gradients
        error_heterogeneity (np.ndarray): 2D array of error values in (tau, gamma) space for omitting axial heterogeneity
        error_threshold (float): threshold value to use for creating the binary decision boundary maps (e.g. 10 for a 10% error threshold)
    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=(6.5, 6))

    ias_gradients_boundary = transform_heatmap(
        error_ias_gradients, threshold=error_threshold
    )
    lateral_gradients_boundary = transform_heatmap(
        error_lateral_gradients, threshold=error_threshold
    )
    heterogeneity_boundary = transform_heatmap(
        error_heterogeneity, threshold=error_threshold
    )

    ias_data = map_data_values(df, error_ias_gradients)
    ias_fraction = 100 * np.sum(ias_data < error_threshold) / len(ias_data)
    print(
        f"Percentage of data points with error from omitting IAS CO2 gradients less than {error_threshold}%: {ias_fraction:.2f}%"
    )

    lateral_data = map_data_values(df, error_lateral_gradients)
    lateral_fraction = 100 * np.sum(lateral_data < error_threshold) / len(lateral_data)
    print(
        f"Percentage of data points with error from omitting lateral gradients less than {error_threshold}%: {lateral_fraction:.2f}%"
    )

    hetero_data = map_data_values(df, error_heterogeneity)
    hetero_fraction = 100 * np.sum(hetero_data < error_threshold) / len(hetero_data)
    print(
        f"Percentage of data points with error from omitting heterogeneity less than {error_threshold}%: {hetero_fraction:.2f}%"
    )

    N = error_ias_gradients.shape[0]
    taus = np.exp(np.linspace(np.log(0.01), np.log(100), N))
    gammas = np.exp(np.linspace(np.log(0.01), np.log(100), N))

    cmap_red, norm = binary_colormap("#b21117ff")  # "#a71413ff"
    cmap_green, _ = binary_colormap("#3d9c5bff")  # "#16440fff"
    cmap_blue, _ = binary_colormap("#206eb3ff")  # "#130a5eff"

    ax.pcolormesh(
        taus, gammas, ias_gradients_boundary, cmap=cmap_red, norm=norm, shading="auto"
    )
    ax.pcolormesh(
        taus,
        gammas,
        lateral_gradients_boundary,
        cmap=cmap_green,
        norm=norm,
        shading="auto",
    )
    ax.pcolormesh(
        taus, gammas, heterogeneity_boundary, cmap=cmap_blue, norm=norm, shading="auto"
    )

    std_frame(ax, df)
    plt.show()
    return
