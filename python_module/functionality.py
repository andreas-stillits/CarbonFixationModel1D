import numpy as np
import pandas as pd

# solve nonlinear equation


def f(gc: float, gias: float, gm_star: float) -> float:
    """
    Get the difference between gm* and its modelled value given gc and gias
    Args:
        gc (float): cellular conductance
        gias (float): intercellular airspace conductance
        gm_star (float): mesophyll conductance at the CO2 compensation point
    Returns:
        float: difference between gm* and its modelled value
    """
    return (
        np.sqrt(np.abs(gc * gias / 2)) * np.tanh(np.sqrt(np.abs(2 * gc / gias)))
        - gm_star
    )


def df_dgc(gc: float, gias: float) -> float:
    """
    Get the derivative of f with respect to gc
    Args:
        gc (float): cellular conductance
        gias (float): intercellular airspace conductance
    Returns:
        float: derivative of f with respect to gc
    """
    return 0.5 * (
        np.sqrt(np.abs(gias / (2 * gc))) * np.tanh(np.sqrt(np.abs(2 * gc / gias)))
        + 1 / (gias * np.cosh(np.sqrt(np.abs(2 * gc / gias))) ** 2)
    )


def newton_solver(
    gc_init: float,
    gias: float,
    gm_star: float,
    step_size: float = 0.4,
    max_iterations: int = 1_000,
    tolerance: float = 1e-6,
) -> float:
    gc = gc_init
    for _ in range(max_iterations):
        f_ = f(gc, gias, gm_star)
        # stop if converged within tolerance
        if abs(f_) < tolerance:
            return np.abs(gc)
        # compute derivative
        df_ = df_dgc(gc, gias)
        # avoid division by zero
        if df_ == 0:
            break
        # update guess
        gc = gc - step_size * f_ / df_
    # error loudly if not converged within max_iterations
    raise ValueError(
        "Newton's method did not converge during gc estimation. "
        "Consider changing step_size, max_iterations, or initial guess."
    )


# -----------------------------------------------------------------------


# estimation helpers


def estimate_Ci_values(
    assimilation_rates: np.ndarray,
    stomatal_conductances: np.ndarray,
    atmospheric_CO2_concentrations: np.ndarray,
) -> np.ndarray:
    """
    Estimate intercellular CO2 concentration (Ci) values from assimilation rates, stomatal conductances, and atmospheric CO2 concentrations.
    Entries must be positive and of equal length. NaN will be assigned to invalid entries.
    Args:
        assimilation_rates (np.ndarray): Array of assimilation rates (An)
        stomatal_conductances (np.ndarray): Array of stomatal conductances (gs)
        atmospheric_CO2_concentrations (np.ndarray): Array of atmospheric CO2 concentrations (Ca
    Returns:
        np.ndarray: Estimated intercellular CO2 concentrations (Ci)
    """

    # verify inputs to be one-dimensional and of equal length
    Ans = assimilation_rates.flatten()
    gss = stomatal_conductances.flatten()
    Cas = atmospheric_CO2_concentrations.flatten()
    if not (Ans.ndim == gss.ndim == Cas.ndim == 1 and len(Ans) == len(gss) == len(Cas)):
        raise ValueError("All inputs must be one-dimensional arrays of equal length.")

    # mask over invalid values (e.g., negative, zero, or NaN)
    valid_mask = (
        (Ans >= 0)
        & (gss > 0)
        & (Cas > 0)
        & (~np.isnan(Ans))
        & (~np.isnan(gss))
        & (~np.isnan(Cas))
    )

    # apply estimation formula
    Cis = np.zeros_like(Ans)
    Cis[valid_mask] = Cas[valid_mask] - (1.6 * Ans[valid_mask] / gss[valid_mask])
    Cis[~valid_mask] = np.nan  # assign NaN to invalid entries
    return Cis


def estimate_gm_star_values(
    assimilation_rates: np.ndarray,
    intercellular_CO2_concentrations: np.ndarray,
    compensation_point_CO2_concentrations: np.ndarray,
) -> np.ndarray:
    """
    Estimate mesophyll conductance at the CO2 compensation point (gm*) values from assimilation rates, intercellular CO2 concentrations, and compensation point CO2 concentrations.
    Entries must be positive and of equal length. NaN will be assigned to invalid entries.
    Args:
        assimilation_rates (np.ndarray): Array of assimilation rates (An)
        intercellular_CO2_concentrations (np.ndarray): Array of intercellular CO2 concentrations (Ci)
        compensation_point_CO2_concentrations (np.ndarray): Array of compensation point CO2 concentrations (C*)
    Returns:
        np.ndarray: Estimated mesophyll conductance at the CO2 compensation point (gm*)
    """
    # verify inputs to be one-dimensional and of equal length
    Ans = assimilation_rates.flatten()
    Cis = intercellular_CO2_concentrations.flatten()
    C_stars = compensation_point_CO2_concentrations.flatten()
    if not (
        Ans.ndim == Cis.ndim == C_stars.ndim == 1
        and len(Ans) == len(Cis) == len(C_stars)
    ):
        raise ValueError("All inputs must be one-dimensional arrays of equal length.")

    # mask over invalid values (e.g., negative, zero, or NaN)
    valid_mask = (
        (Ans >= 0)
        & (Cis > C_stars)
        & (C_stars > 0)
        & (~np.isnan(Ans))
        & (~np.isnan(Cis))
        & (~np.isnan(C_stars))
    )
    # apply estimation formula
    gm_stars = np.zeros_like(Ans)
    gm_stars[valid_mask] = Ans[valid_mask] / (Cis[valid_mask] - C_stars[valid_mask])
    gm_stars[~valid_mask] = np.nan  # assign NaN to invalid entries
    return gm_stars


# -----------------------------------------------------------------------


# numpy entry point


def numpy_estimate_parameters(
    mesophyll_conductances: np.ndarray,
    stomatal_conductances: np.ndarray,
    intercellular_airspace_conductances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate tau and gamma values from mesophyll conductances, stomatal conductances, and intercellular airspace conductances.
    Entries must be positive and of equal length. NaN will be assigned to invalid entries.
    Args:
        mesophyll_conductances (np.ndarray): Array of mesophyll conductances at the compensation point(gm*)
        stomatal_conductances (np.ndarray): Array of stomatal conductances (gs)
        intercellular_airspace_conductances (np.ndarray): Array of intercellular airspace conductances (gias)
    Returns:
        tuple[np.ndarray, np.ndarray]: Estimated tau and gamma values
    """
    # verify inputs to be one-dimensional and of equal length
    gm_stars = mesophyll_conductances.flatten()
    gss = stomatal_conductances.flatten()
    giass = intercellular_airspace_conductances.flatten()
    if not (
        gm_stars.ndim == gss.ndim == giass.ndim == 1
        and len(gm_stars) == len(gss) == len(giass)
    ):
        raise ValueError("All inputs must be one-dimensional arrays of equal length.")

    # mask over invalid values (e.g., negative, zero, or NaN)
    valid_mask = (
        (gm_stars > 0)
        & (gss > 0)
        & (giass > 0)
        & (~np.isnan(gm_stars))
        & (~np.isnan(gss))
        & (~np.isnan(giass))
    )

    # apply estimation formulas
    gammas = np.zeros_like(gm_stars)
    gammas[valid_mask] = 2 * gss[valid_mask] / giass[valid_mask]
    gammas[~valid_mask] = np.nan  # assign NaN to invalid entries

    taus = np.zeros_like(gm_stars)
    for index, valid in enumerate(valid_mask):
        if valid:
            # use Newton's method to estimate tau
            gm_star = gm_stars[index]
            gias = giass[index]
            # initial guess for gc
            gc_init = gm_star
            try:
                gc_estimated = newton_solver(gc_init, gias, gm_star)
                taus[index] = np.sqrt(2 * gc_estimated / gias)
            except ValueError:
                taus[index] = np.nan  # assign NaN if Newton's method fails
        else:
            taus[index] = np.nan  # assign NaN to invalid entries

    return taus, gammas


# ----------------------------------------------------------------------------


# pandas entry point


def require_columns(columns: set[str], df: pd.DataFrame) -> None:
    """Validation helper"""
    if not columns.issubset(df.columns):
        missing = columns - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}")


def pandas_estimate_parameters(
    df: pd.DataFrame, column_map: dict[str, str] | None = None
) -> pd.DataFrame:
    """
    Estimate tau and gamma parameters and add them to the DataFrame.
    The DataFrame must contain 'gs' and 'gias' columns, and either 'gm_star' or
    'An', 'Ci', 'C_star' or 'An', 'Ca', 'C_star' columns.
    Supported column names can be remapped using the optional column_map argument.
    Default expected column names:
        - 'gs': stomatal conductance
        - 'gias': intercellular airspace conductance
        - 'gm_star': mesophyll conductance at the CO2 compensation point
        - 'An': assimilation rate
        - 'Ci': intercellular CO2 concentration
        - 'Ca': atmospheric CO2 concentration
        - 'C_star': CO2 compensation point concentration

    Args:
        df (pd.DataFrame): Input DataFrame with required named columns
        column_map (dict[str, str] | None): Optional mapping of expected column names to DataFrame column names
    Returns:
        pd.DataFrame: DataFrame with added 'tau' and 'gamma' columns

    """
    DEFAULT_COL_MAP = {
        "gs": "gs",
        "gias": "gias",
        "gm_star": "gm_star",
        "An": "An",
        "Ci": "Ci",
        "Ca": "Ca",
        "C_star": "C_star",
    }

    df_ = df.copy()

    # resolve names
    colmap = DEFAULT_COL_MAP.copy()
    if column_map is not None:
        colmap.update(column_map)

    # require stomatal and intercellular airspace conductances to be present
    require_columns({colmap["gs"], colmap["gias"]}, df_)
    gss = df_[colmap["gs"]].to_numpy()
    giass = df_[colmap["gias"]].to_numpy()
    gm_stars = np.zeros_like(gss)

    # branch based on available columns
    cols = set(df_.columns)
    if {colmap["gm_star"]} <= cols:
        gm_stars = df_[colmap["gm_star"]].to_numpy()

    elif {colmap["An"], colmap["Ci"], colmap["C_star"]} <= cols:
        Ans = df_[colmap["An"]].to_numpy()
        Cis = df_[colmap["Ci"]].to_numpy()
        C_stars = df_[colmap["C_star"]].to_numpy()
        gm_stars = estimate_gm_star_values(Ans, Cis, C_stars)

    elif {colmap["An"], colmap["Ca"], colmap["C_star"]} <= cols:
        Ans = df_[colmap["An"]].to_numpy()
        Cas = df_[colmap["Ca"]].to_numpy()
        C_stars = df_[colmap["C_star"]].to_numpy()
        Cis = estimate_Ci_values(Ans, gss, Cas)
        gm_stars = estimate_gm_star_values(Ans, Cis, C_stars)

    else:
        raise ValueError(
            "DataFrame must contain either 'gm_star' or "
            "'An', 'Ci', 'C_star' or 'An', 'Ca', 'C_star' columns."
        )

    # estimate tau and gamma and add to DataFrame
    taus, gammas = numpy_estimate_parameters(gm_stars, gss, giass)
    df_["tau"] = taus
    df_["gamma"] = gammas
    return df_
