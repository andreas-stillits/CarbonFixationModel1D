


import numpy as np 
from argparse import ArgumentParser 
from mpi4py import MPI
from review.utils.mpiscan2d import parallelize


class System():
    def __init__(self, base_level: int) -> None:
        self.base_level = base_level
        
    def ensure_base_path(self) -> None:
        pass  # No file operations in this test

    def get_scan_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        arr1 = np.arange(0, 10, 1) + self.base_level
        arr2 = np.arange(0, 10, 1) + self.base_level
        return arr1, arr2

    def compute_result(self, param1: float, param2: float) -> float:
        return param1 + param2 + self.base_level

    def save_results(self, results: np.ndarray) -> None:
        print(f"Results saved with shape {results.shape} for base level {self.base_level}")
        print(results)



def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Test utils module")
    parser.add_argument("--base", type=int, default=0, help="Base level value")
    args = parser.parse_args(argv)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank == 0:
        print(f"Running with {size} ranks and base level {args.base}")
    parallelize(System(base_level=args.base), comm)
    if rank == 0:
        print("Test completed.")

    return 0 


if __name__ == "__main__":
    raise SystemExit(main())