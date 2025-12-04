"""

Module to scan over temporal variation in atmospheric concentration Ca using MPI

"""

from mpi4py import MPI
import numpy as np
import time

"""
The work to be done must be encoded in a Systems class that exposes                           where to store results
- get_scan_arrays() -> tuple[np.ndarray, np.ndarray]:    the two arrays to scan over
- compute_result(param1: float, param2: float) -> float: compute the result for given parameters
- save_results(results: np.ndarray) -> None:             save the final results array

"""


def parallelize(system, comm: MPI.Intracomm) -> None:
    # initialize MPI communicator info
    rank = comm.Get_rank()
    size = comm.Get_size()
    # start timing on rank 0 
    if rank == 0:
        start_time = time.time()

    # extract data and shapes
    arr1, arr2 = system.get_scan_arrays()
    n1, n2 = arr1.size, arr2.size
    N_total = n1 * n2

    # indices and results
    all_indices = np.arange(N_total)
    local_indices = all_indices[rank::size]
    local_results = []

    for index in local_indices:
        i = index // n2
        j = index % n2
        param1 = arr1[i]
        param2 = arr2[j]
        result = system.compute_result(param1, param2)
        local_results.append((index, result))

    # gather results on rank 0
    gathered_results = comm.gather(local_results, root=0)

    if rank == 0:
        flat_results = np.empty(N_total, dtype=float)
        for chunk in gathered_results:
            for index, result in chunk:
                flat_results[index] = result
        results = flat_results.reshape((n1, n2))
        system.save_results(results)
        end_time = time.time()
        print(f"2D scanning completed in {(end_time - start_time):.2f} seconds")
