"""

Scan over (tau, gamma) space to reproduce figure 3C for exponential profiles

"""


from review.nonlinear.solver import NonlinearSolver
from review.utils.constants import NonlinearExploration
from review.utils.profiles import StepProfile
from review.utils.mpiscan2d import parallelize 
from mpi4py import MPI 
import numpy as np
from argparse import ArgumentParser

class System():
    def __init__(self, constants: NonlinearExploration, mu: float) -> None:
        self.constants = constants
        self.base_path = constants.get_base_path()
        self.mu = mu

    def get_scan_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        taus = self.constants.get_tau_range()
        gammas = self.constants.get_gamma_range()
        return taus, gammas

    def compute_result(self, tau: float, gamma: float) -> float:
        c = self.constants # alias
        params = (tau, gamma, c.chi_)
        mu = self.mu 
        rho_kappas = c.get_rho_kappa_range()
        rho_deltas = c.get_rho_delta_range()
        rho_lambdas = c.get_rho_lambda_range()
        # get homogeneous non-linear solution
        solver = NonlinearSolver(params, mu)
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
                    solver = NonlinearSolver(params, mu, delta=delta.steadify(), kappa=kappa.steadify())
                    domain, solution = solver.solve()
                    del solver
                    drawdown = 1 - solution[0]
                    relative_drawdowns[count] = (drawdown_hom - drawdown) / drawdown_hom
                    count += 1
        result = np.sqrt(np.sum(relative_drawdowns**2)/c.n_rho**3)
        return result
        
    def save_results(self, results: np.ndarray) -> None:
        filename = self.base_path / self.constants.get_filename(self.mu)
        np.savetxt(filename, results, delimiter=self.constants.delimiter)


def main(argv: list[str] | None = None) -> int: 
    parser = ArgumentParser(description="MPI scan over nonlinear parameters for figure 3C")
    parser.add_argument("--mu", type=float, default=0.0, help="Value of mu to compute")
    args = parser.parse_args(argv)
    #
    comm = MPI.COMM_WORLD 
    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank == 0:
        print(f"Running with {size} rank(s) and mu={args.mu} for nonlinear exploration of figure 3C", flush=True)
    parallelize(System(NonlinearExploration(), args.mu), comm)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

