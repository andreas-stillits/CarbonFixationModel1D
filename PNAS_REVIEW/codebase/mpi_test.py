""" 

Conduct a small MPI test to ensure it works correctly.

"""


from mpi4py import MPI

import time


def main():

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    all_params = [i for i in range(8)]
    if rank == 0:
        print(f"Distributing {all_params} among {size} processes.")

    local_params = all_params[rank::size]
    print(f"Process {rank} received parameters: {local_params}")

    local_results = [param ** 2 for param in local_params]
    print(f"Process {rank} computed results: {local_results}")

    gathered = comm.gather(local_results, root=0)
    if rank == 0:
        print(f"Gathered results at root: {gathered}")
        print(f"flattening...")
        flattened = [item for sublist in gathered for item in sublist]
        print(f"Flattened results: {flattened}")


if __name__ == "__main__":
    raise SystemExit(main())