"""

Module for solving a simple 3d continuous cylindrical problem wit discrete stomata

"""

import numpy as np
from mpi4py import MPI
import adios4dolfinx as a4x
from pathlib import Path
import ufl
from dolfinx import fem, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import gmshio
from review.utils.constants import ThreeDimExploration, Cases

TOLERANCE = 1e-6
DEFAULT_QDEGREE = 8
DEFAULT_RESOLUTION = 50


def get_delta(
    x: ufl.SpatialCoordinate, rho: tuple[float, float, float], epsilon: float = 0.01
):
    rho_delta, rho_kappa, rho_lambda = rho
    maximum = 1 / ((1 - rho_lambda) + rho_lambda * rho_delta)
    minimum = rho_delta * maximum
    offset = 1 - rho_lambda
    delta = minimum + (maximum - minimum) * 0.5 * (
        1 - ufl.tanh((x[2] - offset) / epsilon)
    )
    return delta


def get_kappa(
    x: ufl.SpatialCoordinate, rho: tuple[float, float, float], epsilon: float = 0.01
):
    rho_delta, rho_kappa, rho_lambda = rho
    maximum = 1 / ((1 - rho_lambda) * rho_kappa + rho_lambda)
    minimum = rho_kappa * maximum
    offset = 1 - rho_lambda
    kappa = minimum + (maximum - minimum) * 0.5 * (
        1 + ufl.tanh((x[2] - offset) / epsilon)
    )
    return kappa


class Steady3DSolver:
    def __init__(
        self,
        params: tuple[float, float, float],  # tau, gamma, chi_
        plug_radius: float,
        stomatal_radius: float,
        mesh_file: str | Path,
        stomatal_blur: float = 0.002,
        filename: str | Path = "../files/3d/steady3d.bp",
        order: int = 1,
        rho: tuple[float, float, float] = (1.0, 1.0, 0.6),
        extract_profile: bool = True,
    ) -> None:
        self.tau = params[0]
        self.gamma = params[1]
        self.chi_ = params[2]
        self.plug_radius = plug_radius
        self.stomatal_radius = stomatal_radius
        self.mesh_file = mesh_file if isinstance(mesh_file, Path) else Path(mesh_file)
        self.stomatal_blur = stomatal_blur
        self.filename = filename if isinstance(filename, Path) else Path(filename)
        self.order = order
        self.rho = rho
        self.extract_profile = extract_profile

    def solve(self) -> None:
        mesh, cell_tags, facet_tags = gmshio.read_from_msh(
            self.mesh_file, MPI.COMM_SELF, 0, gdim=3
        )
        #
        AIRSPACE_TAG = 1
        # TOP_TAG = 2
        BOTTOM_TAG = 3
        # CURVED_TAG = 4
        #
        functionspace = fem.functionspace(mesh, ("Lagrange", self.order))
        dx = ufl.Measure("dx", domain=mesh, subdomain_data=cell_tags)
        ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)

        tau2 = fem.Constant(mesh, default_scalar_type(self.tau**2))
        chi_ = fem.Constant(mesh, default_scalar_type(self.chi_))

        chi = ufl.TrialFunction(functionspace)
        v = ufl.TestFunction(functionspace)
        x = ufl.SpatialCoordinate(mesh)
        #
        phi = x[0] ** 2 + x[1] ** 2 - self.stomatal_radius**2  # type: ignore[reportIndexIssue]
        gamma = (
            self.gamma
            * (self.plug_radius / self.stomatal_radius) ** 2
            * 0.5
            * (1 - ufl.tanh(phi / self.stomatal_blur / self.plug_radius**2))
        )
        #
        delta = get_delta(x, self.rho, epsilon=0.01)
        kappa = get_kappa(x, self.rho, epsilon=0.01)
        # Weak form
        a = (
            delta * ufl.inner(ufl.grad(chi), ufl.grad(v)) * dx(AIRSPACE_TAG)
            + tau2 * kappa * chi * v * dx(AIRSPACE_TAG)
            + gamma * chi * v * ds(BOTTOM_TAG)
        )

        L = tau2 * kappa * chi_ * v * dx(AIRSPACE_TAG) + gamma * v * ds(BOTTOM_TAG)

        problem = LinearProblem(
            a, L, bcs=[], petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
        )
        chi_h = problem.solve()
        # save solution
        a4x.write_mesh(self.filename, mesh)
        a4x.write_meshtags(self.filename, mesh, cell_tags, meshtag_name="cell_tags")
        a4x.write_meshtags(self.filename, mesh, facet_tags, meshtag_name="facet_tags")
        a4x.write_function(self.filename, chi_h, name="solution")
        #
        plug_area = np.pi * self.plug_radius**2
        an3d = (
            fem.assemble_scalar(fem.form(tau2 * kappa * (chi_h - chi_) * dx))
            / plug_area
        )
        if not self.extract_profile:
            # calculate assimilation rate and return
            return an3d, [], [], []
        else:
            # extract z profile and calculate assimilation rate

            # load in solution
            dx = ufl.Measure(
                "dx",
                domain=mesh,
                subdomain_data=cell_tags,
                metadata={"quadrature_degree": DEFAULT_QDEGREE},
            )
            # extract z profile of mean and variance
            zmin, zmax = mesh.geometry.x[:, 2].min(), mesh.geometry.x[:, 2].max()
            dz = (zmax - zmin) / DEFAULT_RESOLUTION
            edges = np.arange(zmin, zmax + TOLERANCE, dz)
            centers = (edges[:-1] + edges[1:]) / 2

            z = ufl.SpatialCoordinate(mesh)[2]

            def get_slice_quantities(a, b):
                inside = ufl.conditional(ufl.And(ufl.ge(z, a), ufl.lt(z, b)), 1.0, 0.0)
                V_solid = fem.assemble_scalar(fem.form(inside * dx))
                U_int = fem.assemble_scalar(fem.form(chi_h * inside * dx))
                U2_int = fem.assemble_scalar(fem.form(chi_h**2 * inside * dx))
                u_avg = U_int / V_solid if V_solid > 0 else np.nan
                u2_avg = U2_int / V_solid if V_solid > 0 else np.nan
                return V_solid, u_avg, u2_avg

            quantities = np.vstack(
                [
                    list(get_slice_quantities(a, b))
                    for a, b in zip(edges[:-1], edges[1:])
                ]
            )

            u_means = quantities[:, 1]
            u2_means = quantities[:, 2]
            u_std = np.sqrt(u2_means - u_means**2)

            return an3d, centers, u_means, u_std
