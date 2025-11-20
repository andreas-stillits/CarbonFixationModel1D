"""

Module to scan over temporal variation in atmospheric concentration Ca using MPI

"""

from mpi4py import MPI
import numpy as np
import time
import functools
from pathlib import Path 
from argparse import ArgumentParser
from review.steady.solver import SteadySolver
from review.utils.constants import SteadyConstants
from review.utils.homogeneous import homogeneous_solution  
from review.utils.constants import add_steady_scanning_flags
from review.utils.profiles import ExponentialProfile
from review.utils.paths import ensure_steady_scanning_paths, get_steady_scanning_path



def main(argv=None) -> int:
    parser = ArgumentParser(description="MPI Steady Scanner over exponential profiles")
    add_steady_scanning_flags(parser)
    args = parser.parse_args(argv)
    #
    sc = SteadyConstants() 
    taus = np.exp(np.linspace(np.log(sc.tau_min), np.log(sc.tau_max), sc.n_tau))
    gammas = np.exp(np.linspace(np.log(sc.gamma_min), np.log(sc.gamma_max), sc.n_gamma))
    N_total = sc.n_tau * sc.n_gamma 
    #
    betas = np.linspace(sc.beta_min, sc.beta_max, sc.n_beta)
    #
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank == 0:
        # Start timer
        start_time = time.time()
        ensure_steady_scanning_paths()
    
    all_indices = np.arange(N_total)
    local_indices = all_indices[rank::size]

    local_results = []
    for index in local_indices:
        i = index // sc.n_gamma
        j = index % sc.n_gamma
        tau = taus[i]
        gamma = gammas[j]
        params = [tau, gamma, sc.chi_]
        drawdown_hom = 1 - homogeneous_solution(0.0, params)
        relative_drawdowns = np.zeros(sc.n_beta**2)
        count = 0
        for idx1 in range(sc.n_beta):
            for idx2 in range(sc.n_beta):
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
        local_results.append((index, np.sqrt(np.sum(relative_drawdowns**2)/sc.n_beta**2)))

    gathered_results = comm.gather(local_results, root=0)

    if rank == 0:
        flat_results = np.empty(N_total, dtype=float)
        for chunk in gathered_results:
            for index, result in chunk:
                flat_results[index] = result
        results = flat_results.reshape((sc.n_tau, sc.n_gamma))
        np.savetxt(get_steady_scanning_path(), results, delimiter=sc.delimiter)

        end_time = time.time()
        print(f"Steady scanning completed in {end_time - start_time:.2f} seconds")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

