"""

Module to scan over temporal variation in atmospheric concentration Ca using MPI

"""

from mpi4py import MPI
import numpy as np
import time
import functools
from pathlib import Path 
from argparse import ArgumentParser
from review.temporal.solver import TemporalSolver 
from review.steady.solver import SteadySolver
from review.utils.homogeneous import homogeneous_solution  
from PNAS_REVIEW.review.utils.constants__ import TemporalConstants, fixed_delta, fixed_kappa, add_temporal_scanning_flags
from review.utils.profiles import OscillatorProfile
from review.utils.paths import ensure_temporal_scanning_paths, get_temporal_scanning_path



def main(argv=None) -> int:
    parser = ArgumentParser(description="MPI Temporal Ca Scanner")
    add_temporal_scanning_flags(parser)
    args = parser.parse_args(argv)
    #
    tc = TemporalConstants() 
    params = tc.get_case_params(args.case)
    amplitude_range = np.exp(np.linspace(np.log(tc.amp_min), np.log(tc.amp_max), tc.n_amp))
    period_range = np.exp(np.linspace(np.log(tc.period_min), np.log(tc.period_max), tc.n_period))
    N_total = tc.n_amp * tc.n_period
    #
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank == 0:
        # Start timer
        start_time = time.time()
        ensure_temporal_scanning_paths()
    
    all_indices = np.arange(N_total)
    local_indices = all_indices[rank::size]

    local_results = []
    for index in local_indices:
        i = index // tc.n_period
        j = index % tc.n_period
        amplitude = amplitude_range[i]
        period = period_range[j]
        #
        osc = OscillatorProfile(amplitude, period)
        timing = (0.0, tc.periods_to_run * period, tc.fraction_of_period * period)
        if args.quantity == "Ca":
            solver = TemporalSolver(params,
                                    timing=timing,
                                    animate=False,
                                    update_delta=fixed_delta,
                                    update_kappa=fixed_kappa,
                                    update_atmospheric=osc.generalize())
        elif args.quantity == "gs":
            solver = TemporalSolver(params,
                                    timing=timing,
                                    animate=False,
                                    update_delta=fixed_delta,
                                    update_kappa=fixed_kappa,
                                    update_stomata=osc.generalize())
        elif args.quantity == "K":
            def osc_kappa(x: np.ndarray, t: float) -> np.ndarray:
                return fixed_kappa(x, t) * osc.generalize()(x, t)
            solver = TemporalSolver(params,
                                    timing=timing,
                                    animate=False,
                                    update_delta=fixed_delta,
                                    update_kappa=osc_kappa)
        times, alphas = solver.solve()
        del solver
        cutoff_index  = int(len(alphas) * tc.periods_to_cut / tc.periods_to_run)
        times, alphas = times[cutoff_index:], alphas[cutoff_index:]
        #
        if args.save_series:
            series_path = get_temporal_scanning_path(args.case, args.quantity) / "timeseries"
            series_path.mkdir(parents=True, exist_ok=True)
            series_path /= f"index{index}.txt"
            np.savetxt(series_path, np.vstack((times, alphas)).T, delimiter=tc.delimiter)
        result = np.mean(alphas)
        local_results.append((index, result))

    gathered_results = comm.gather(local_results, root=0)

    if rank == 0:
        path = get_temporal_scanning_path(args.case, args.quantity)
        flat_results = np.empty(N_total, dtype=float)
        for chunk in gathered_results:
            for index, result in chunk:
                flat_results[index] = result
        results = flat_results.reshape((tc.n_amp, tc.n_period))
        header = f"params {params[0]}, {params[1]}, {params[2]}, amplitude_range ({tc.amp_min},{tc.amp_max},{tc.n_amp}), period_range ({tc.period_min},{tc.period_max},{tc.n_period})"
        np.savetxt(path / "mean_alpha.txt", results, delimiter=tc.delimiter, header=header)
        #
        alpha_hom = params[1] * (1 - homogeneous_solution(0, params))
        np.savetxt(path / "alpha_hom.txt", np.array([alpha_hom]), delimiter=tc.delimiter, header=f"params {params[0]}, {params[1]}, {params[2]}")
        #
        if args.quantity == "K":
            solver = SteadySolver(params, 
                                  display=True, 
                                  display_name=path / "steady_chi.png", 
                                  delta=functools.partial(fixed_delta, t=0), kappa=functools.partial(fixed_kappa, t=0))
            domain, solution = solver.solve()
            alpha_het = params[1] * (1 - solution[0])
            np.savetxt(path / "alpha_het.txt", np.array([alpha_het]), delimiter=tc.delimiter, header=f"params {params[0]}, {params[1]}, {params[2]}")

        end_time = time.time()
        print(f"Total computation time for case{args.case} - {args.quantity} : {end_time - start_time:.2f} seconds")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

