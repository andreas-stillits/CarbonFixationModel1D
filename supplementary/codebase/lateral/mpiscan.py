"""

Scan over (tau, gamma) space to reproduce figure 3C for 3D model

"""

from codebase.lateral.solver import Steady3DSolver
from codebase.utils.constants import ThreeDimExploration
from codebase.utils.mpiscan2d import parallelize
from codebase.utils import paths
from mpi4py import MPI
import numpy as np
from argparse import ArgumentParser


class System:
    def __init__(
        self, constants: ThreeDimExploration, contrast: float, version: str
    ) -> None:
        self.constants = constants
        self.base_path = paths.get_base_path() / "lateral_scanning"
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.contrast = contrast
        self.version = version

    def get_scan_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        taus = self.constants.get_tau_range()
        gammas = self.constants.get_gamma_range()
        return taus, gammas

    def compute_result(self, tau: float, gamma: float) -> float:
        c = self.constants  # alias
        params = (tau, gamma, c.chi_)
        #
        plug_radius = c.get_plug_radius(version=self.version)
        stomatal_radius = c.get_stomatal_radius(version=self.version)
        mesh_file = c.get_mesh_path(version=self.version)

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

        # get heterogeneous 3D solution
        solver = Steady3DSolver(
            params,
            plug_radius,
            stomatal_radius,
            mesh_file,
            stomatal_blur=c.stomatal_epsilon,
            rho=(self.contrast, self.contrast, 0.5),
            extract_profile=False,
            save_solution=False,
        )
        an3d_het, _, _, _ = solver.solve()
        del solver

        result = np.abs(an3d_hom - an3d_het) / an3d_het
        return result

    def save_results(self, results: np.ndarray) -> None:
        filedir = self.base_path / "heterogeneity_scan"
        filedir.mkdir(parents=True, exist_ok=True)
        filename = filedir / f"3D_{self.version}_contrast_{self.contrast:.2f}.txt"
        np.savetxt(filename, results, delimiter=";")


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(
        description="MPI scan over nonlinear parameters for figure 3C"
    )
    parser.add_argument(
        "version",
        choices=["low", "typical", "high"],
        help="Version of plug aspect ratio, choose from 'low', 'typical', 'high'",
    )
    parser.add_argument(
        "--contrast",
        "-c",
        type=float,
        default=0.5,
        help="Value of contrast to compute for",
    )
    args = parser.parse_args(argv)
    #
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank == 0:
        print(
            f"Running with {size} rank(s), version={args.version}, and contrast={args.contrast} for 3D exploration of figure 3C",
            flush=True,
        )
    parallelize(System(ThreeDimExploration(), args.contrast, args.version), comm)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
