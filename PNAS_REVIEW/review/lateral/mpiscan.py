"""

Scan over (tau, gamma) space to reproduce figure 3C for 3D model

"""

from review.lateral.solver import Steady3DSolver
from review.utils.constants import ThreeDimExploration
from review.utils.mpiscan2d import parallelize
from mpi4py import MPI
import numpy as np
from argparse import ArgumentParser


class System:
    def __init__(self, constants: ThreeDimExploration, rhomax: float) -> None:
        self.constants = constants
        self.base_path = constants.get_base_path("typical").parent
        self.rhomax = rhomax

    def get_scan_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        taus = self.constants.get_tau_range()
        gammas = self.constants.get_gamma_range()
        return taus, gammas

    def compute_result(self, tau: float, gamma: float) -> float:
        c = self.constants  # alias
        params = (tau, gamma, c.chi_)
        rho_kappas = c.get_rho_kappa_range(self.rhomax)
        rho_deltas = c.get_rho_delta_range(self.rhomax)
        rho_lambdas = c.get_rho_lambda_range(self.rhomax)
        #
        plug_radius = c.get_plug_radius(version="typical")
        stomatal_radius = c.get_stomatal_radius(version="typical")
        mesh_file = c.get_mesh_path(version="typical")

        # get homogeneous 3D solution
        solver = Steady3DSolver(
            params,
            plug_radius,
            stomatal_radius,
            mesh_file,
            stomatal_blur=c.stomatal_epsilon,
            rho=(1.0, 1.0, 0.5),
            extract_profile=False,
            save_solution=False,
        )
        an3d_hom, _, _, _ = solver.solve()
        del solver
        relative_drawdowns = np.zeros(c.n_rho**3)
        count = 0
        for idx1 in range(c.n_rho):
            for idx2 in range(c.n_rho):
                for idx3 in range(c.n_rho):
                    rho_kappa = rho_kappas[idx1]
                    rho_delta = rho_deltas[idx2]
                    rho_lambda = rho_lambdas[idx3]
                    rho = (rho_delta, rho_kappa, rho_lambda)
                    solver = Steady3DSolver(
                        params,
                        plug_radius,
                        stomatal_radius,
                        mesh_file,
                        stomatal_blur=c.stomatal_epsilon,
                        rho=rho,
                        extract_profile=False,
                        save_solution=False,
                    )
                    an3d, _, _, _ = solver.solve()
                    del solver
                    assert an3d_hom > 0.0, f"Unphysical solution for rho = {rho}"
                    relative_drawdowns[count] = (an3d_hom - an3d) / an3d
                    count += 1
        result = np.sqrt(np.sum(relative_drawdowns**2) / c.n_rho**3)
        return result

    def save_results(self, results: np.ndarray) -> None:
        filedir = self.base_path / f"rhomax_{self.rhomax:.1f}_"
        filedir.mkdir(parents=True, exist_ok=True)
        filename = filedir / "3d_sensitivities.txt"
        np.savetxt(filename, results, delimiter=self.constants.delimiter)


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(
        description="MPI scan over nonlinear parameters for figure 3C"
    )
    parser.add_argument(
        "--rhomax", type=float, default=0.2, help="Value of rhomax to compute"
    )
    args = parser.parse_args(argv)
    #
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank == 0:
        print(
            f"Running with {size} rank(s) and rhomax={args.rhomax} for 3D exploration of figure 3C",
            flush=True,
        )
    parallelize(System(ThreeDimExploration(), args.rhomax), comm)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
