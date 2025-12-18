"""

Scan over (tau, gamma) space to reproduce figure 3C for exponential profiles

"""

from codebase.temporal.solver import TemporalSolver
from codebase.utils.constants import TemporalExploration, Cases
from codebase.utils.profiles import OscillatorProfile
from codebase.utils.mpiscan2d import parallelize
from codebase.utils import paths
from mpi4py import MPI
import numpy as np
from argparse import ArgumentParser


class System:
    def __init__(
        self, constants: TemporalExploration, case: str, quantity: str
    ) -> None:
        self.constants = constants
        self.params = Cases().get_case_params(case)
        self.case = case
        self.quantity = quantity
        self.base_path = (
            paths.get_base_path()
            / "temporal_scanning"
            / f"rhomax_{constants.rho[0]:.1f}_"
            / f"{quantity}"
        )
        self.base_path.mkdir(parents=True, exist_ok=True)

    def get_scan_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        period_range = self.constants.get_period_range()
        amplitude_range = self.constants.get_amplitude_range()
        return period_range, amplitude_range

    # helper function for readibility
    def get_solvers(
        self, period: float, amplitude: float
    ) -> tuple[TemporalSolver, TemporalSolver]:
        params = Cases().get_case_params(self.case)
        timing = self.constants.get_timing(period)
        oscillator = OscillatorProfile(amplitude, period).generalize()
        delta = self.constants.get_fixed_delta  # callable (x, t)
        kappa = self.constants.get_fixed_kappa  # callable (x, t)
        # Switch on quantities
        if self.quantity == "Ca":
            solver_hom = TemporalSolver(
                params, timing=timing, update_atmospheric=oscillator
            )
            solver_het = TemporalSolver(
                params,
                timing=timing,
                update_atmospheric=oscillator,
                update_delta=delta,
                update_kappa=kappa,
            )
        elif self.quantity == "gs":
            solver_hom = TemporalSolver(
                params, timing=timing, update_stomata=oscillator
            )
            solver_het = TemporalSolver(
                params,
                timing=timing,
                update_stomata=oscillator,
                update_delta=delta,
                update_kappa=kappa,
            )
        elif self.quantity == "K":
            solver_hom = TemporalSolver(params, timing=timing, update_kappa=oscillator)

            def kappa_oscillator(x: np.ndarray, t: float) -> np.ndarray:
                return kappa(x, t) * oscillator(x, t)

            solver_het = TemporalSolver(
                params, timing=timing, update_delta=delta, update_kappa=kappa_oscillator
            )
        else:
            raise ValueError(f"Unknown quantity: {self.quantity}")
        return solver_hom, solver_het

    def compute_result(self, period: float, amplitude: float) -> float:
        c = self.constants
        solver_hom, solver_het = self.get_solvers(period, amplitude)
        domain_hom, sol_hom = solver_hom.solve()
        domain_het, sol_het = solver_het.solve()
        del solver_hom
        del solver_het
        assert len(sol_hom) == len(sol_het), "solutions have different shapes"
        # cut transients
        cut_index = int(len(domain_hom) * c.periods_to_cut / c.periods_to_run)
        sol_hom = sol_hom[cut_index:]
        sol_het = sol_het[cut_index:]
        # # save timeseries for debugging
        # filename = self.base_path / self.case
        # filename.mkdir(exist_ok=True, parents=True)
        # np.savetxt(filename / f"timeseries_amp{amplitude:.2f}_per{period:.2f}_hom.txt", sol_hom, delimiter=self.constants.delimiter)
        # np.savetxt(filename / f"timeseries_amp{amplitude:.2f}_per{period:.2f}_het.txt", sol_het, delimiter=self.constants.delimiter)

        # calculate local relative difference and integrate over time
        result = np.sqrt(np.mean(((sol_hom - sol_het) / sol_het) ** 2))
        return result

    def save_results(self, results: np.ndarray) -> None:
        filename = self.base_path / f"variations_{self.case}.txt"
        np.savetxt(filename, results, delimiter=";")


def main(argv: list[str] | None = None) -> int:
    constants = TemporalExploration()
    parser = ArgumentParser(
        description="MPI scan over nonlinear parameters for figure 3C"
    )
    parser.add_argument(
        "quantity", type=str, choices=["Ca", "gs", "K"], help="quantity to drive"
    )
    parser.add_argument(
        "case",
        type=str,
        choices=["A", "B", "C", "D", "E"],
        help="tau, gamma case to explore",
    )
    parser.add_argument(
        "--rhomax",
        type=float,
        default=0.0,
        help="maximum rho value for delta/kappa profiles. If not given program will run with defaults from review.utils.constants.TemporalExploration and write to /tmp/",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=constants.n_period,
        help="number of points in each dimension (period, amplitude)",
    )
    args = parser.parse_args(argv)
    #
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank == 0:
        print(
            f"Running with {size} rank(s) and ({args.quantity},{args.case}) for temporal exploration",
            flush=True,
        )
    if args.rhomax > 0.0:
        rho = constants.rho
        constants.rho = (args.rhomax, args.rhomax, rho[2])
    constants.n_amp = args.resolution
    constants.n_period = args.resolution
    parallelize(System(constants, args.case, args.quantity), comm)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
