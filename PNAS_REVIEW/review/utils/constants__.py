""" 

Module to hold global constants across variations of Ca, gs, K

"""

from dataclasses import dataclass 
from argparse import ArgumentParser
from review.utils.profiles import StepProfile
import numpy as np
from typing import Callable

GLOBAL_RHO = (0.5, 0.6, 0.6)
GLOBAL_EPSILON = 0.01

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

@dataclass(frozen=True)
class SteadyConstants:

    beta_min: float = 0.3
    beta_max: float = 3.0
    n_beta: int = 5 

    tau_min: float = 0.01
    tau_max: float = 100.0
    n_tau: int = 50

    gamma_min: float = 0.01
    gamma_max: float = 100.0
    n_gamma: int = 50

    chi_: float = 0.1

    delimiter: str = ";"


def fixed_delta(x: np.ndarray, t: float) -> np.ndarray:
    step = StepProfile(epsilon=GLOBAL_EPSILON, direction="down")
    step.populate_rho(GLOBAL_RHO[0], GLOBAL_RHO[2])
    return step.generalize()(x, t)

def fixed_kappa(x: np.ndarray, t: float) -> np.ndarray:
    step = StepProfile(epsilon=GLOBAL_EPSILON, direction="up")
    step.populate_rho(GLOBAL_RHO[1], GLOBAL_RHO[2])
    return step.generalize()(x, t)

def add_temporal_scanning_flags(parser: ArgumentParser) -> None: 
    parser.add_argument("--case", type=str, choices=["A", "B", "C", "D", "E"], help="Case selection for parameters")
    parser.add_argument("--quantity", type=str, choices=["Ca", "gs", "K"], help="Type of parameter scan")
    parser.add_argument("--save-series", action="store_true", help="Flag to save time series data")


def add_steady_scanning_flags(parser: ArgumentParser) -> None: 
    pass
