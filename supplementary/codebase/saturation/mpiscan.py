from codebase.saturation.solver import SaturatedSolver
from codebase.utils.constants import NonlinearExploration
from codebase.utils.profiles import StepProfile
from codebase.utils.mpiscan2d import parallelize
from codebase.utils import paths
from mpi4py import MPI
import numpy as np
from argparse import ArgumentParser


class System:
    def __init__(self, constants: NonlinearExploration) -> None:
        self.constants = constants
        self.base_path = paths.get_base_path() / "nonlinear_sensitivities"
        self.base_path.mkdir(parents=True, exist_ok=True)

    def get_scan_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        taus = self.constants.get_tau_range()
        gammas = self.constants.get_gamma_range()
        return taus, gammas

    def compute_result(self, tau: float, gamma: float) -> float:
        c = self.constants  # alias
        params = (tau, gamma, c.chi_)
        rho_kappas = c.get_rho_kappa_range()
        rho_deltas = c.get_rho_delta_range()
        rho_lambdas = c.get_rho_lambda_range()
        # get homogeneous non-linear solution
        solver = SaturatedSolver(params)
        domain, homogeneous_solution = solver.solve()
        del solver
        drawdown_hom = 1 - homogeneous_solution[0]
        relative_drawdowns = np.zeros(c.n_rho**3)
        count = 0
        for idx1 in range(c.n_rho):
            for idx2 in range(c.n_rho):
                for idx3 in range(c.n_rho):
                    rho_kappa = rho_kappas[idx1]
                    rho_delta = rho_deltas[idx2]
                    rho_lambda = rho_lambdas[idx3]
                    delta = StepProfile(direction="down")
                    delta.populate_rho(rho_delta, rho_lambda)
                    kappa = StepProfile(direction="up")
                    kappa.populate_rho(rho_kappa, rho_lambda)
                    solver = SaturatedSolver(
                        params, delta=delta.steadify(), kappa=kappa.steadify()
                    )
                    domain, solution = solver.solve()
                    del solver
                    drawdown = 1 - solution[0]
                    relative_drawdowns[count] = (drawdown_hom - drawdown) / drawdown_hom
                    count += 1
        result = np.sqrt(np.sum(relative_drawdowns**2) / c.n_rho**3)
        return result

    def save_results(self, results: np.ndarray) -> None:
        filename = self.base_path / "sensitivities_muInf_.txt"
        np.savetxt(filename, results, delimiter=";")


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(
        description="MPI scan over saturated parameters for figure 3C"
    )
    args = parser.parse_args(argv)
    #
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank == 0:
        print(
            f"Running with {size} rank(s) for saturated exploration of figure 3C",
            flush=True,
        )
    parallelize(System(NonlinearExploration()), comm)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
