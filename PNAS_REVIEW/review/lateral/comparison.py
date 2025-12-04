"""

Extract lateral mean and variance from 3d solution.
Save informative plot: (z, C +- C_err) profile
Summarize as relative difference between assimilation rate predictions

"""

from argparse import ArgumentParser
from review.utils.constants import ThreeDimExploration, Cases
from review.utils.homogeneous import homogeneous_solution
from mpi4py import MPI
import adios4dolfinx as a4x
from dolfinx import fem
import ufl
import numpy as np

TOLERANCE = 1e-6
DEFAULT_QDEGREE = 8
DEFAULT_RESOLUTION = 25


def main(argv: list[str] | None = None) -> int:
    # parse case and version
    parser = ArgumentParser(description="Process 3D lateral solutions.")
    parser.add_argument(
        "version", choices=["min", "mean", "max"], help="Version to process"
    )
    parser.add_argument(
        "case", choices=["A", "B", "C", "D", "E"], help="Case to process"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=DEFAULT_RESOLUTION,
        help="Number of bins along z-axis",
    )
    args = parser.parse_args(argv)

    # setup
    cases = Cases()
    constants = ThreeDimExploration()
    fileroot = str(constants.get_base_path(args.version) / f"solution{args.case}")

    # load in solution
    mesh = a4x.read_mesh(fileroot + ".bp", MPI.COMM_WORLD)
    cell_tags = a4x.read_meshtags(fileroot + ".bp", mesh, meshtag_name="cell_tags")
    facet_tags = a4x.read_meshtags(fileroot + ".bp", mesh, meshtag_name="facet_tags")

    dx = ufl.Measure(
        "dx",
        domain=mesh,
        subdomain_data=cell_tags,
        metadata={"quadrature_degree": DEFAULT_QDEGREE},
    )
    functionspace = fem.functionspace(mesh, ("Lagrange", 1))
    chi = fem.Function(functionspace)
    a4x.read_function(fileroot + ".bp", chi, name="solution")

    # extract z profile of mean and variance
    zmin, zmax = mesh.geometry.x[:, 2].min(), mesh.geometry.x[:, 2].max()
    dz = (zmax - zmin) / args.resolution
    edges = np.arange(zmin, zmax + TOLERANCE, dz)
    centers = (edges[:-1] + edges[1:]) / 2

    z = ufl.SpatialCoordinate(mesh)[2]

    def get_slice_quantities(a, b):
        inside = ufl.conditional(ufl.And(ufl.ge(z, a), ufl.lt(z, b)), 1.0, 0.0)
        V_solid = fem.assemble_scalar(fem.form(inside * dx))
        U_int = fem.assemble_scalar(fem.form(chi * inside * dx))
        U2_int = fem.assemble_scalar(fem.form(chi**2 * inside * dx))
        u_avg = U_int / V_solid if V_solid > 0 else np.nan
        u2_avg = U2_int / V_solid if V_solid > 0 else np.nan
        return V_solid, u_avg, u2_avg

    quantities = np.vstack(
        [list(get_slice_quantities(a, b)) for a, b in zip(edges[:-1], edges[1:])]
    )

    V_solids = quantities[:, 0]
    u_means = quantities[:, 1]
    u2_means = quantities[:, 2]
    u_std = np.sqrt(u2_means - u_means**2)

    # save data
    data = np.vstack([centers, V_solids, u_means, u_std]).T
    np.savetxt(fileroot + ".txt", data, header="z V_solid u_mean u_std", delimiter=";")

    # save an3d, an1d and relative difference
    tau, gamma, chi_ = cases.get_case_params(args.case)
    an3d = fem.assemble_scalar(fem.form(tau**2 * (chi - chi_) * dx)) / (
        np.mean(V_solids) / dz
    )
    an1d = gamma * (1 - homogeneous_solution(0, (tau, gamma, chi_)))
    rel_diff = (an1d - an3d) / an3d
    np.savetxt(
        fileroot + "_summary.txt",
        np.array([[an3d, an1d, rel_diff]]),
        header="A_n_3D A_n_1D rel_diff",
        delimiter=";",
    )

    print(f"{fem.assemble_scalar(fem.form(chi * dx)) / np.sum(V_solids):.2f}")
    print(f"{np.sum(V_solids):.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
