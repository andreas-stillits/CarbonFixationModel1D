"""

Module to hold keep all global choices in one place

"""

from dataclasses import dataclass
from review.utils.profiles import StepProfile
from review.utils import paths
import numpy as np
from pathlib import Path


def _get_log_range(minimum: float, maximum: float, n_points: int) -> np.ndarray:
    return np.exp(np.linspace(np.log(minimum), np.log(maximum), n_points))


@dataclass
class Cases:
    caseA: tuple[float, float, float] = (0.10, 10.0, 0.1)  # tau, gamma, chi_
    caseB: tuple[float, float, float] = (0.31, 0.10, 0.1)
    caseC: tuple[float, float, float] = (10.0, 0.10, 0.1)
    caseD: tuple[float, float, float] = (1.00, 1.00, 0.1)
    caseE: tuple[float, float, float] = (3.16, 10.0, 0.1)

    def get_case_params(self, case: str) -> tuple[float, float, float]:
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


@dataclass
class TemporalExploration:
    amp_min: float = 0.01
    amp_max: float = 0.5
    n_amp: int = 7

    period_min: float = 0.02
    period_max: float = 20.0
    n_period: int = 9

    periods_to_run: int = 15
    periods_to_cut: int = 10
    fraction_of_period: float = 0.05

    # fixed delta, kappa profiles
    epsilon: float = 0.02
    rho: tuple[float, float, float] = (0.5, 0.5, 0.6)

    delimiter: str = ";"

    foldername: str = "temporal_scanning/tmp/"

    def get_amplitude_range(self) -> np.ndarray:
        return np.linspace(self.amp_min, self.amp_max, self.n_amp)

    def get_period_range(self) -> np.ndarray:
        return _get_log_range(self.period_min, self.period_max, self.n_period)

    def get_timing(self, period: float) -> tuple[float, float, float]:
        return (0.0, self.periods_to_run * period, self.fraction_of_period * period)

    def get_fixed_delta(self, x: np.ndarray, t: float) -> np.ndarray:
        step = StepProfile(epsilon=self.epsilon, direction="down")
        step.populate_rho(self.rho[0], self.rho[2])
        return step.generalize()(x, t)

    def get_fixed_kappa(self, x: np.ndarray, t: float) -> np.ndarray:
        step = StepProfile(epsilon=self.epsilon, direction="up")
        step.populate_rho(self.rho[1], self.rho[2])
        return step.generalize()(x, t)

    def get_base_path(self, case: str, quantity: str) -> Path:
        base_path = paths.get_base_path(ensure=True) / self.foldername / quantity
        base_path.mkdir(parents=True, exist_ok=True)
        return base_path


@dataclass
class ReproduceWithExponentials:
    beta_min: float = 0.25
    beta_max: float = 2.5
    n_beta: int = 5

    tau_min: float = 0.01
    tau_max: float = 100.0
    n_tau: int = 100

    gamma_min: float = 0.01
    gamma_max: float = 100.0
    n_gamma: int = 100

    chi_: float = 0.1

    delimiter: str = ";"

    foldername: str = "exponential_sensitivities"
    filename: str = "sensitivities.txt"

    def get_beta_range(self) -> np.ndarray:
        return np.linspace(self.beta_min, self.beta_max, self.n_beta)

    def get_tau_range(self) -> np.ndarray:
        return _get_log_range(self.tau_min, self.tau_max, self.n_tau)

    def get_gamma_range(self) -> np.ndarray:
        return _get_log_range(self.gamma_min, self.gamma_max, self.n_gamma)

    def get_base_path(self) -> Path:
        base_path = paths.get_base_path(ensure=True) / self.foldername
        base_path.mkdir(parents=True, exist_ok=True)
        return base_path


@dataclass
class NonlinearExploration:
    rho_delta_min: float = 0.2
    rho_kappa_min: float = 0.2
    rho_lambda_min: float = 0.2
    n_rho: int = 3

    tau_min: float = 0.01
    tau_max: float = 100.0
    n_tau: int = 100

    gamma_min: float = 0.01
    gamma_max: float = 100.0
    n_gamma: int = 100

    chi_: float = 0.1

    delimiter: str = ";"
    foldername: str = "nonlinear_sensitivities"

    def get_rho_kappa_range(self) -> np.ndarray:
        return np.linspace(self.rho_kappa_min, 1.0, self.n_rho)

    def get_rho_delta_range(self) -> np.ndarray:
        return np.linspace(self.rho_delta_min, 1.0, self.n_rho)

    def get_rho_lambda_range(self) -> np.ndarray:
        return np.linspace(self.rho_lambda_min, 1.0 - self.rho_lambda_min, self.n_rho)

    def get_tau_range(self) -> np.ndarray:
        return _get_log_range(self.tau_min, self.tau_max, self.n_tau)

    def get_gamma_range(self) -> np.ndarray:
        return _get_log_range(self.gamma_min, self.gamma_max, self.n_gamma)

    def get_base_path(self) -> Path:
        base_path = paths.get_base_path(ensure=True) / self.foldername
        base_path.mkdir(parents=True, exist_ok=True)
        return base_path

    def get_filename(self, mu: float) -> str:
        return f"sensitivities_mu{mu:.2f}_.txt"


@dataclass
class ThreeDimExploration:
    # units of µm
    stomatal_spacing_min: float = 19.88
    stomatal_spacing_mean: float = 36.32
    stomatal_spacing_max: float = 167.32
    # units of µm
    mesophyll_thickness_min: float = 20.60
    mesophyll_thickness_mean: float = 238.75
    mesophyll_thickness_max: float = 809.00
    # unit of µm
    stomatal_pore_radius: float = (
        4.0  # typical, bound in range 2-6 µm (when normalizing to a circle)
    )
    allowed = ("low", "typical", "high")
    # rescaled quantities
    factor = 2.0
    plug_radius_typical: float = stomatal_spacing_mean / mesophyll_thickness_mean
    plug_radius_low: float = plug_radius_typical / factor
    plug_radius_high: float = plug_radius_typical * factor

    stomatal_radius_typical: float = stomatal_pore_radius / mesophyll_thickness_mean
    stomatal_radius_low: float = stomatal_radius_typical / factor
    stomatal_radius_high: float = stomatal_radius_typical * factor

    stomatal_epsilon: float = 0.00015
    foldername: str = "lateral_scanning"
    delimiter: str = ";"

    def get_base_path(self, version: str) -> Path:
        assert version in self.allowed, f"Unknown version: {version}"
        base_path = paths.get_base_path(ensure=True) / self.foldername / version
        base_path.mkdir(parents=True, exist_ok=True)
        return base_path

    def get_mesh_path(self, version: str) -> Path:
        if version in self.allowed:
            mesh_path = (
                paths.get_base_path(ensure=True) / "meshes" / f"cylinder_{version}.msh"
            )
            return mesh_path
        else:
            raise ValueError(f"Unknown version: {version}")

    def get_plug_radius(self, version: str) -> float:
        if version == "low":
            return self.plug_radius_low
        elif version == "typical":
            return self.plug_radius_typical
        elif version == "high":
            return self.plug_radius_high
        else:
            raise ValueError(f"Unknown version: {version}")

    def get_stomatal_radius(self, version: str) -> float:
        if version == "low":
            return self.stomatal_radius_low
        elif version == "typical":
            return self.stomatal_radius_typical
        elif version == "high":
            return self.stomatal_radius_high
        else:
            raise ValueError(f"Unknown version: {version}")
