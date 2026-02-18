"""

Scan over (tau, gamma) space to get 3D vs 1D/0D error

"""

from codebase.lateral.solver import Steady3DSolver
from codebase.utils.constants import ThreeDimExploration
from codebase.utils.mpiscan2d import parallelize
from codebase.utils import paths
from mpi4py import MPI
import numpy as np
from argparse import ArgumentParser

from codebase.utils.homogeneous import homogeneous_solution

VERSION = "typical"


class System:
    def __init__(
        self, constants: ThreeDimExploration, resolution: int, dim: str
    ) -> None:
        self.constants = constants
        self.base_path = paths.get_base_path() / "lateral_scanning"
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.resolution = resolution
        self.dim = dim

    def get_scan_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        taus = np.exp(
            np.linspace(
                np.log(self.constants.tau_min),
                np.log(self.constants.tau_max),
                self.resolution,
            )
        )
        gammas = np.exp(
            np.linspace(
                np.log(self.constants.gamma_min),
                np.log(self.constants.gamma_max),
                self.resolution,
            )
        )
        return taus, gammas

    def compute_result(self, tau: float, gamma: float) -> float:
        c = self.constants  # alias
        params = (tau, gamma, c.chi_)
        #
        plug_radius = c.get_plug_radius(version=VERSION)
        stomatal_radius = c.get_stomatal_radius(version=VERSION)
        mesh_file = c.get_mesh_path(version=VERSION)

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
        an3d, chi_i, _, _ = solver.solve()
        assert an3d > 0.0, "3D assimilation rate should be positive"
        del solver

        if self.dim == "1D":
            # 1D comparison
            an1d = params[1] * (1 - homogeneous_solution(0, params))
            relative_difference = abs(an1d - an3d) / abs(an3d)
            return relative_difference
        else:
            # 0D comparison
            an0d = params[0] ** 2 * (chi_i - params[2])
            relative_difference = abs(an0d - an3d) / abs(an3d)
            return relative_difference

    def save_results(self, results: np.ndarray) -> None:
        filedir = self.base_path / "fine_grained"
        filedir.mkdir(parents=True, exist_ok=True)
        filename = filedir / f"typical_3Dv{self.dim}_error.txt"
        np.savetxt(filename, results, delimiter=";")


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(
        description="Scan over (tau, gamma) space to get 3D vs 1D error for typical parameters."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=10,
        help="Number of points per axis in the scan",
    )
    parser.add_argument(
        "--dim",
        choices=["1D", "0D"],
        default="1D",
        help="Whether to compare 3D solution to 1D or 0D solution",
    )
    args = parser.parse_args(argv)
    #
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank == 0:
        print(
            f"Running with {size} rank(s) for exploration of typical 3D vs {args.dim} error",
            flush=True,
        )
    parallelize(System(ThreeDimExploration(), args.resolution, args.dim), comm)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
