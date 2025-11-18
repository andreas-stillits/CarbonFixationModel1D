"""

Module to scan over temporal variation in atmospheric concentration Ca using MPI

"""

from mpi4py import MPI
import numpy as np
import time
from PNAS_REVIEW.codebase.temporal.temporal_solver import TemporalSolver, get_homogeneous_solution
import functools
from constants import TemporalConstants, general_oscillator, add_argparse_flags, general_delta, general_kappa
from argparse import ArgumentParser
import os
import sys 
sys.path.append("../steady_state")
import ss_solver, utils


def get_local_result_Ca(params: list[float], amplitude: float, period: float, tc: TemporalConstants) -> tuple[np.ndarray, np.ndarray]:
    update_osc = functools.partial(general_oscillator, 
                                   amplitude=amplitude, 
                                   period=period)
    timing = (0.0, tc.periods_to_run * period, tc.fraction_of_period * period)
    solver = TemporalSolver(params,
                            timing=timing,
                            animate=False,
                            update_delta=general_delta,
                            update_kappa=general_kappa,
                            update_atmospheric=update_osc)
    times, alphas = solver.solve()
    del solver
    cutoff_index = int(len(alphas) * tc.periods_to_cut / tc.periods_to_run)
    return times[cutoff_index:], alphas[cutoff_index:]

def get_local_result_gs(params: list[float], amplitude: float, period: float, tc: TemporalConstants) -> tuple[np.ndarray, np.ndarray]:
    update_osc = functools.partial(general_oscillator, 
                                   amplitude=amplitude, 
                                   period=period)    
    timing = (0.0, tc.periods_to_run * period, tc.fraction_of_period * period)
    solver = TemporalSolver(params,
                            timing=timing,
                            animate=False,
                            update_delta=general_delta,
                            update_kappa=general_kappa,
                            update_stomata=update_osc)
    times, alphas = solver.solve()
    del solver
    cutoff_index = int(len(alphas) * tc.periods_to_cut / tc.periods_to_run)
    return times[cutoff_index:], alphas[cutoff_index:]


def get_local_result_K(params: list[float], amplitude: float, period: float, tc: TemporalConstants) -> tuple[np.ndarray, np.ndarray]:
    update_osc = functools.partial(general_oscillator, 
                                   amplitude=amplitude, 
                                   period=period)
    timing = (0.0, tc.periods_to_run * period, tc.fraction_of_period * period)
    def osc_kappa(x: np.ndarray, t: float) -> np.ndarray:
        return general_kappa(x, t) * update_osc(x, t)
    
    solver = TemporalSolver(params,
                            timing=timing,
                            animate=False,
                            update_delta=general_delta,
                            update_kappa=osc_kappa)
    times, alphas = solver.solve()
    del solver
    cutoff_index = int(len(alphas) * tc.periods_to_cut / tc.periods_to_run)
    return times[cutoff_index:], alphas[cutoff_index:]


def main(argv=None) -> int:
    parser = ArgumentParser(description="MPI Temporal Ca Scanner")
    add_argparse_flags(parser)
    args = parser.parse_args(argv)
    parent_path = f"showcases/case{args.case}/{args.quantity}/"
    os.makedirs(parent_path, exist_ok=True)
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
    
    all_indices = np.arange(N_total)
    local_indices = all_indices[rank::size]

    local_results = []
    for index in local_indices:
        i = index // tc.n_period
        j = index % tc.n_period
        amplitude = amplitude_range[i]
        period = period_range[j]
        #
        if args.quantity == "Ca":
            times, alphas = get_local_result_Ca(params, amplitude, period, tc)
        elif args.quantity == "gs":
            times, alphas = get_local_result_gs(params, amplitude, period, tc)
        elif args.quantity == "K":
            times, alphas = get_local_result_K(params, amplitude, period, tc)
        #
        if args.save_series:
            series_path = f"{parent_path}/timeseries/"
            os.makedirs(series_path, exist_ok=True)
            series_path += f"index{index}.txt"
            np.savetxt(series_path, np.vstack((times, alphas)).T, delimiter=tc.delimiter)
        result = np.mean(alphas)
        local_results.append((index, result))

    gathered_results = comm.gather(local_results, root=0)

    if rank == 0:
        flat_results = np.empty(N_total, dtype=float)
        for chunk in gathered_results:
            for index, result in chunk:
                flat_results[index] = result
        
        results = flat_results.reshape((tc.n_amp, tc.n_period))
        header = f"params {params[0]}, {params[1]}, {params[2]}, amplitude_range ({tc.amp_min},{tc.amp_max},{tc.n_amp}), period_range ({tc.period_min},{tc.period_max},{tc.n_period})"
        np.savetxt(f"showcases/case{args.case}/{args.quantity}/mean_alpha.txt", results, delimiter=tc.delimiter, header=header)
        #
        alpha_hom = params[1] * (1 - get_homogeneous_solution(0, params))
        np.savetxt(f"showcases/case{args.case}/{args.quantity}/alpha_hom.txt", np.array([alpha_hom]), delimiter=tc.delimiter, header=f"params {params[0]}, {params[1]}, {params[2]}")
        #
        if args.quantity == "K":
            mesh, uh = ss_solver.solver(params, delta=functools.partial(general_delta, t=0), kappa=functools.partial(general_kappa, t=0), save=False)
            domain, solution = utils.extract_solution_from_objects(mesh, uh)
            alpha_het = params[1] * (1 - solution[0])
            np.savetxt(f"showcases/case{args.case}/{args.quantity}/alpha_het.txt", np.array([alpha_het]), delimiter=tc.delimiter, header=f"params {params[0]}, {params[1]}, {params[2]}")


        end_time = time.time()
        print(f"Total computation time for case{args.case} - {args.quantity} : {end_time - start_time:.2f} seconds")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

