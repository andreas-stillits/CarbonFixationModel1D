import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from core import (
    estimate_Ci_values,
    estimate_gm_star_values,
    newton_solver,
)


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
