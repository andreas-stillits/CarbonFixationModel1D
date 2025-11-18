""" 

Module for solving steady state problems.

"""

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import adios4dolfinx as a4x
import ufl 
from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem
from typing import Callable


class SteadySolver:
    def __init__(self,
                 params: list[float],
                 domain: mesh.Mesh | None = None,
                 domain_resolution: int = 100,
                 functionspace: fem.FunctionSpace | None = None,
                 display: bool = False,
                 display_name: str = "chi_steady.png",
                 order: int = 1,
                 delta: Callable[[np.ndarray], np.ndarray] | None = None,
                 kappa: Callable[[np.ndarray], np.ndarray] | None = None):
        self.tau = params[0]
        self.gamma = params[1]
        self.chi_ = params[2]
        self.domain = domain
        self.domain_resolution = domain_resolution
        self.functionspace = functionspace
        self.display = display
        self.display_name = display_name
        self.order = order
        self.delta = (lambda x: np.ones_like(x)) if delta is None else delta
        self.kappa = (lambda x: np.ones_like(x)) if kappa is None else kappa

    def setup_domain(self) -> tuple[ufl.Measure, ufl.Measure]:
        if self.domain is None:
            self.domain = mesh.create_interval(MPI.COMM_SELF, self.domain_resolution, [0.0, 1.0])
        if self.functionspace is None:
            self.functionspace = fem.functionspace(self.domain, ("CG", self.order))
        # --- Create facet tags --- 
        tdim = self.domain.topology.dim 
        fdim = tdim - 1
        facets_left = mesh.locate_entities(self.domain, fdim, lambda x: np.isclose(x[0], 0.0))
        facets_right = mesh.locate_entities(self.domain, fdim, lambda x: np.isclose(x[0], 1.0))
        facet_indices = np.concatenate([facets_left, facets_right])
        facet_tags = np.concatenate([np.full_like(facets_left, 1), np.full_like(facets_right, 2)])
        mt = mesh.meshtags(self.domain, fdim, facet_indices, facet_tags)
        dx = ufl.Measure("dx", domain=self.domain)
        ds = ufl.Measure("ds", domain=self.domain, subdomain_data=mt)
        return dx, ds 
    
    def solve(self) -> tuple[np.ndarray, np.ndarray]:
        # --- get measures and set up domain mesh and functionspace if not supplied ---
        dx, ds = self.setup_domain()
        
        # --- simulation parameters ---
        tau2  = fem.Constant(self.domain, PETSc.ScalarType(self.tau**2))
        gamma = fem.Constant(self.domain, PETSc.ScalarType(self.gamma))
        chi_  = fem.Constant(self.domain, PETSc.ScalarType(self.chi_))
        
        # --- Trial and Test Functions ---
        chi = ufl.TrialFunction(self.functionspace)
        v   = ufl.TestFunction(self.functionspace)
        x   = self.functionspace.tabulate_dof_coordinates()[:, 0]

        delta = fem.Function(self.functionspace, name="delta")
        delta.x.array[:] = self.delta(x)
        
        kappa = fem.Function(self.functionspace, name="kappa")
        kappa.x.array[:] = self.kappa(x)

        # build forms
        a = - delta * ufl.inner(ufl.grad(chi), ufl.grad(v)) * dx \
            - tau2 * kappa * chi * v * dx \
            - gamma * chi * v * ds(1)
        
        L = - tau2 * kappa * chi_ * v * dx \
            - gamma * v * ds(1)
        
        problem = LinearProblem(a, L, bcs=[], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        uh = problem.solve()   

        # extract numpy domain and solution
        self.domain.topology.create_connectivity(0, 1)
        imap = self.domain.topology.index_map(0)
        nloc_vertices = imap.size_local 
        local_vertices = np.arange(nloc_vertices)

        v_dofs = fem.locate_dofs_topological(self.functionspace, 0, local_vertices)
        xv = self.domain.geometry.x[:nloc_vertices, 0].copy()
        uv = uh.x.array[v_dofs].copy()
        seq = np.argsort(xv)
        np_domain = xv[seq]
        np_solution = uv[seq]
        if self.display:
            np.savetxt(self.display_name, np.vstack([np_domain, np_solution]).T, delimiter=";")
        
        return np_domain, np_solution











def solver(params: list[float], delta: Callable, kappa: Callable, domain: mesh.Mesh | None = None, output_path: str = "../files/run.bp", save: bool = False, domain_resolution: int = 100):
    """ 
    Solve the steady-state model for given parameters, delta(z), kappa(z)
    
    Parameters
    ----------
    params : list[float]
        List of parameters [tau, gamma, chi_]
    delta : callable
        Function mapping spatial coordinate to delta(z)
    kappa : callable
        Function mapping spatial coordinate to kappa(z)
    domain: mesh.Mesh | None
        The mesh domain to solve on. If None, a default mesh will be used.
    output_path : str
        Path to save the output files
    save : bool
        Whether to save the solution to file
    """
    tau, gamma, chi_ = params
    if domain is None:
        domain = mesh.create_interval(MPI.COMM_WORLD, domain_resolution, [0.0, 1.0])
    V = fem.functionspace(domain, ("CG", 1))

    # Create facet tags 
    tdim = domain.topology.dim 
    fdim = tdim - 1
    facets_left = mesh.locate_entities(domain, fdim, lambda x: np.isclose(x[0], 0.0))
    facets_right = mesh.locate_entities(domain, fdim, lambda x: np.isclose(x[0], 1.0))
    facet_indices = np.concatenate([facets_left, facets_right])
    facet_tags = np.concatenate([np.full_like(facets_left, 1), np.full_like(facets_right, 2)])
    mt = mesh.meshtags(domain, fdim, facet_indices, facet_tags)
    dx = ufl.Measure("dx", domain=domain)
    ds = ufl.Measure("ds", domain=domain, subdomain_data=mt)

    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)
    tau_ = fem.Constant(domain, PETSc.ScalarType(tau))
    gamma_ = fem.Constant(domain, PETSc.ScalarType(gamma))
    chi_star = fem.Constant(domain, PETSc.ScalarType(chi_))

    # build forms
    a = - delta(x) * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx \
        - tau_**2 * kappa(x) * u * v * dx \
        - gamma_ * u * v * ds(1)
    
    L = - tau_**2 * kappa(x) * chi_star * v * dx \
        - gamma_ * v * ds(1)

    problem = LinearProblem(a, L, bcs=[], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()

    if save:
        # save solution
        a4x.write_mesh(output_path, domain)
        a4x.write_function(output_path, uh, name="CO2_profile")
    return domain, uh 

