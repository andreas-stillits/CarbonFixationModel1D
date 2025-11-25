""" 

Module to hold keep all global choices in one place

"""

from dataclasses import dataclass 
from argparse import ArgumentParser
from review.utils.profiles import StepProfile
import numpy as np

def _get_log_range(minimum: float, maximum: float, n_points: int) -> np.ndarray:
    return np.exp(np.linspace(np.log(minimum), np.log(maximum), n_points))

@dataclass(frozen=True)
class Cases:
    caseA: list[float] = [0.10, 10.0, 0.1]  # tau, gamma, chi_
    caseB: list[float] = [0.31, 0.10, 0.1]
    caseC: list[float] = [10.0, 0.10, 0.1]
    caseD: list[float] = [1.00, 1.00, 0.1]
    caseE: list[float] = [3.16, 10.0, 0.1]

    def get_case_params(self, case: str) -> list[float]:
        if case == "A":
            return self.caseA
        elif case == "B":
            return self.caseB
        elif case == "C":
            return self.caseC
        elif case == "D":
            return self.caseD
        elif case == "E":
            return self.caseE
        else:
            raise ValueError(f"Unknown case: {case}")
        

@dataclass(frozen=True)
class TemporalExploration:
    amp_min: float = 0.01
    amp_max: float = 0.5
    n_amp: int = 4

    period_min: float = 0.1
    period_max: float = 10.0
    n_period: int = 8

    periods_to_run: int = 20
    periods_to_cut: int = 10
    fraction_of_period: float = 0.05

    # fixed delta, kappa profiles
    epsilon: float = 0.02
    rho: tuple[float, float, float] = (0.5, 0.6, 0.6)

    delimiter: str = ";"

    def get_amplitude_range(self) -> np.ndarray:
        return _get_log_range(self.amp_min, self.amp_max, self.n_amp)
    
    def get_period_range(self) -> np.ndarray:
        return _get_log_range(self.period_min, self.period_max, self.n_period)

    def get_timing(self) -> tuple[float, float, float]:
        return (0.0, self.periods_to_run*self.period, self.fraction_of_period*self.period)
    
    def get_fixed_delta(self,x: np.ndarray, t: float) -> np.ndarray:
        step = StepProfile(epsilon=self.epsilon, direction="down")
        step.populate_rho(self.rho[0], self.rho[2])
        return step.generalize()(x, t)    

    def get_fixed_kappa(self, x: np.ndarray, t: float) -> np.ndarray:
        step = StepProfile(epsilon=self.epsilon, direction="up")
        step.populate_rho(self.rho[1], self.rho[2])
        return step.generalize()(x, t)   


@dataclass(frozen=True)
class ReproduceWithExponentials:
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

    def get_beta_range(self) -> np.ndarray:
        return _get_log_range(self.beta_min, self.beta_max, self.n_beta)

    def get_tau_range(self) -> np.ndarray:
        return _get_log_range(self.tau_min, self.tau_max, self.n_tau)

    def get_gamma_range(self) -> np.ndarray:
        return _get_log_range(self.gamma_min, self.gamma_max, self.n_gamma)


@dataclass(frozen=True)
class NonlinearExploration:
    mu_min: float = 0.0
    mu_max: float = 3.0
    n_mu: int = 10

    rho_delta_min: float = 0.4
    rho_kappa_min: float = 0.4
    rho_lambda_min: float = 0.2
    n_rho: int = 5

    tau_min: float = 0.01
    tau_max: float = 100.0
    n_tau: int = 50

    gamma_min: float = 0.01
    gamma_max: float = 100.0
    n_gamma: int = 50

    chi_: float = 0.1

    delimiter: str = ";"

    def get_mu_range(self) -> np.ndarray:
        return _get_log_range(self.mu_min, self.mu_max, self.n_mu)

    def get_rho_kappa_range(self) -> np.ndarray:
        return _get_log_range(self.rho_kappa_min, 1.0, self.n_rho)

    def get_rho_delta_range(self) -> np.ndarray:
        return _get_log_range(self.rho_delta_min, 1.0, self.n_rho)
    
    def get_rho_lambda_range(self) -> np.ndarray:
        return _get_log_range(self.rho_lambda_min, 1.0 - self.rho_lambda_min, self.n_rho)

    def get_tau_range(self) -> np.ndarray:
        return _get_log_range(self.tau_min, self.tau_max, self.n_tau)

    def get_gamma_range(self) -> np.ndarray:
        return _get_log_range(self.gamma_min, self.gamma_max, self.n_gamma)
    

@dataclass(frozen=True)
class ThreeDimExploration:
    stomatal_ratio_min: float = 0.02
    stomatal_ratio_max: float = 1.00

    aspect_ratio_min: float   = 0.05
    aspect_ratio_max: float   = 0.30

    delimiter: str = ";"



