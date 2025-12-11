"""

Extract lateral mean and variance from 3d solution.
Save informative plot: (z, C +- C_err) profile
Summarize as relative difference between assimilation rate predictions

"""

from argparse import ArgumentParser
from pathlib import Path
from review.utils.constants import ThreeDimExploration, Cases
from review.utils.homogeneous import homogeneous_solution
from review.lateral.solver import Steady3DSolver
import numpy as np


def perform_dimensional_comparison(
    params: tuple[float, float, float],
    plug_radius: float,
    stomatal_radius: float,
    mesh_file: str | Path,
    stomatal_blur: float = 0.002,
    filename_3d: str | Path = "../files/3d/steady3d.bp",
    filename_profile: str | Path = "../files/3d/steady3d_profile.txt",
    filename_summary: str | Path = "../files/3d/steady3d_summary.txt",
    rho: tuple[float, float, float] = (1.0, 1.0, 0.6),
) -> tuple[float, np.ndarray]:
    solver = Steady3DSolver(
        params=params,
        plug_radius=plug_radius,
        stomatal_radius=stomatal_radius,
        mesh_file=mesh_file,
        stomatal_blur=stomatal_blur,
        filename=filename_3d,
        rho=rho,
        extract_profile=True,
        save_solution=True,
    )
    an3d, z, chi_mean, chi_std = solver.solve()
    profile = np.vstack((z, chi_mean, chi_std)).T
    an1d = params[1] * (1 - homogeneous_solution(0, params))
    assert an3d > 0.0, "3D assimilation rate should be positive"
    relative_difference = abs(an1d - an3d) / abs(an3d)
    # save data
    np.savetxt(filename_profile, profile, header="z; chi_mean; chi_std", delimiter=";")
    np.savetxt(
        filename_summary,
        np.array([an3d, an1d, relative_difference]),
        header="an3d; an1d; rel_diff",
        delimiter=";",
    )
    return relative_difference, profile


def main(argv: list[str] | None = None) -> int:

    parser = ArgumentParser(
        description="Compare 3D lateral solver against 1D homogeneous solution."
    )
    parser.add_argument("version", choices=["low", "typical", "high"])
    parser.add_argument("case", choices=["A", "B", "C", "D", "E"])
    args = parser.parse_args(argv)

    cases = Cases()
    constants = ThreeDimExploration()

    fileroot = constants.get_base_path(args.version)
    filename_3d = fileroot / f"solution{args.case}.bp"
    filename_profile = fileroot / f"solution{args.case}_profile.txt"
    filename_summary = fileroot / f"solution{args.case}_summary.txt"

    _ = perform_dimensional_comparison(
        cases.get_case_params(args.case),
        constants.get_plug_radius(args.version),
        constants.get_stomatal_radius(args.version),
        constants.get_mesh_path(args.version),
        stomatal_blur=constants.stomatal_epsilon,
        filename_3d=filename_3d,
        filename_profile=filename_profile,
        filename_summary=filename_summary,
        rho=(1.0, 1.0, 0.6),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
