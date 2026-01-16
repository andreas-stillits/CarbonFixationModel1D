import numpy as np


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
