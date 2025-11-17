"""

Module to compute parameter scan over MPI processes.

"""

from mpi4py import MPI
import numpy as np
import time 
from solver import TemporalSolver, get_homogeneous_solution
import functools

params = [10.0, 10.0, 0.1] # tau, gamma, chi_ 
amp_min, amp_max, n_amp = 0.01, 0.5, 4 
period_min, period_max, n_period = 0.1, 10.0, 8


amplitude_range = np.exp(np.linspace(np.log(amp_min), np.log(amp_max), n_amp))
period_range = np.exp(np.linspace(np.log(period_min), np.log(period_max), n_period))
N_total = n_amp * n_period

periods_to_run = 20 
periods_to_cut = 10
fraction_of_period = 0.05

alpha_ss = params[1]* (1 - get_homogeneous_solution(0, params))


def update_osc_general(x: np.ndarray, t: float, amplitude: float, period: float) -> np.ndarray:
    return 1 + amplitude * np.sin(2 * np.pi / period * t)

def get_local_result(amplitude: float, period: float) -> float:
    update_osc = functools.partial(update_osc_general, 
                                   amplitude=amplitude, 
                                   period=period)
    timing = (0.0, periods_to_run * period, fraction_of_period * period)
    solver = TemporalSolver(params,
                            timing=timing,
                            animate=False,
                            update_atmospheric=update_osc)
    times, alphas = solver.solve()
    del solver
    cutoff_index = int(len(alphas) * periods_to_cut / periods_to_run)
    return times[cutoff_index:], alphas[cutoff_index:]


def main():
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
        i = index // n_period
        j = index % n_period
        amplitude = amplitude_range[i]
        period = period_range[j]

        times, alphas = get_local_result(amplitude, period)
        # write to file 
        np.savetxt(f"timeseries/mpi_output_index{index}.txt", 
                   np.vstack([times, alphas]).T, 
                   delimiter=";")
        result = (np.mean(alphas) - alpha_ss)/alpha_ss
        local_results.append((index, result))

    gathered = comm.gather(local_results, root=0)

    if rank == 0:
        flat_results = np.empty(N_total, dtype=float)
        for chunk in gathered:
            for index, result in chunk:
                flat_results[index] = result 
        
        results = flat_results.reshape((n_amp, n_period))
        np.savetxt("results.txt", results, delimiter=";", header=f"params {params[0]}, {params[1]}, {params[2]}, amplitude_range ({amp_min},{amp_max},{n_amp}), period_range ({period_min},{period_max},{n_period})")
    
        print(amplitude_range)
        print(period_range)
        print(results)

        end_time = time.time()
        print(f"Total computation time: {end_time - start_time:.2f} seconds")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())