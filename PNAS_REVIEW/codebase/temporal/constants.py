""" 

Module to hold global constants across variations of Ca, gs, K

"""

from dataclasses import dataclass 
from argparse import ArgumentParser
import numpy as np

GLOBAL_RHO = (0.5, 0.6, 0.6)


@dataclass(frozen=True)
class TemporalConstants:

    amp_min: float = 0.01
    amp_max: float = 0.5
    n_amp: int = 4

    period_min: float = 0.1
    period_max: float = 10.0
    n_period: int = 8

    periods_to_run: int = 20
    periods_to_cut: int = 10
    fraction_of_period: float = 0.05

    delimiter: str = ";"

    def get_case_params(self, case: str) -> list[float]:
        if case == "A":
            return [0.10, 10.0, 0.1]  # tau, gamma, chi_
        elif case == "B":
            return [0.31, 0.10, 0.1]
        elif case == "C":
            return [10.0, 0.10, 0.1]
        elif case == "D":
            return [1.00, 1.00, 0.1]
        elif case == "E":
            return [3.16, 10.0, 0.1]
        else:
            raise ValueError(f"Unknown case: {case}")


def add_argparse_flags(parser: ArgumentParser) -> None: 
    parser.add_argument("--case", type=str, choices=["A", "B", "C", "D", "E"], help="Case selection for parameters")
    parser.add_argument("--quantity", type=str, choices=["Ca", "gs", "K"], help="Type of parameter scan")
    parser.add_argument("--save-series", action="store_true", help="Flag to save time series data")


def general_oscillator(x: np.ndarray, t: float, amplitude: float, period: float) -> np.ndarray:
    return 1 + amplitude * np.sin(2 * np.pi / period * t)

def general_delta(x: np.ndarray, t: float) -> np.ndarray:
    rho_delta, rho_kappa, rho_lambda = GLOBAL_RHO
    delta_max = 1/((1-rho_lambda) + rho_lambda*rho_delta)
    delta_min = rho_delta*delta_max
    offset    = 1 - rho_lambda
    epsilon   = 0.01
    return delta_min + (delta_max - delta_min) * 0.5 * (1 - np.tanh((x - offset) / epsilon))

def general_kappa(x: np.ndarray, t: float) -> np.ndarray:
    rho_delta, rho_kappa, rho_lambda = GLOBAL_RHO
    kappa_max = 1/((1-rho_lambda)*rho_kappa + rho_lambda)
    kappa_min = rho_kappa*kappa_max
    offset    = 1 - rho_lambda
    epsilon   = 0.01
    return kappa_min + (kappa_max - kappa_min) * 0.5 * (1 + np.tanh((x - offset) / epsilon))

