"""

Scan over (tau, gamma) space to reproduce figure 3C for exponential profiles

"""

from codebase.steady.solver import SteadySolver
from codebase.utils.constants import ReproduceWithExponentials
from codebase.utils.homogeneous import homogeneous_solution
from codebase.utils.profiles import ExponentialProfile
from codebase.utils.mpiscan2d import parallelize
from mpi4py import MPI
import numpy as np


class System:
    def __init__(self, constants: ReproduceWithExponentials) -> None:
        self.constants = constants
        self.base_path = constants.get_base_path()

    def get_scan_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        taus = self.constants.get_tau_range()
        gammas = self.constants.get_gamma_range()
        return taus, gammas

    def compute_result(self, tau: float, gamma: float) -> float:
        c = self.constants  # alias
        params = (tau, gamma, c.chi_)
        betas = c.get_beta_range()
        drawdown_hom = 1 - homogeneous_solution(0.0, params)  # type: ignore[reportArgumentType]
        relative_drawdowns = np.zeros(c.n_beta**2)
        count = 0
        for idx1 in range(c.n_beta):
            for idx2 in range(c.n_beta):
                beta_delta = betas[idx1]
                beta_kappa = betas[idx2]
                delta = ExponentialProfile(beta_delta, direction="down").steadify()
                kappa = ExponentialProfile(beta_kappa, direction="up").steadify()
                solver = SteadySolver(params, delta=delta, kappa=kappa)
                _, solution = solver.solve()
                del solver
                drawdown = 1 - solution[0]
                relative_drawdowns[count] = (drawdown_hom - drawdown) / drawdown_hom
                count += 1
        result = np.sqrt(np.sum(relative_drawdowns**2) / c.n_beta**2)
        return result

    def save_results(self, results: np.ndarray) -> None:
        filename = self.base_path / self.constants.filename
        np.savetxt(filename, results, delimiter=self.constants.delimiter)


def main() -> int:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank == 0:
        print(
            f"Running with {size} rank(s) for reproducing figure 3C with exponentials",
            flush=True,
        )
    parallelize(System(ReproduceWithExponentials()), comm)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
