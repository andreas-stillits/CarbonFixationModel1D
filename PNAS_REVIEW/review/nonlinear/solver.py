import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl 
from dolfinx import fem, mesh, nls
from dolfinx.fem.petsc import NonlinearProblem
import matplotlib.pyplot as plt 
from typing import Callable


class NonlinearSolver: 
    def __init__(self, 
                params: list[float], # tau, gamma, chi_
                mu: float = 0.0,
                domain: mesh.Mesh | None = None,
                domain_resolution: int = 100,
                functionspace: fem.FunctionSpace | None = None,
                display: bool = False,
                display_name: str = "chi_nonlinear.png",
                order: int = 1,
                delta: Callable[[np.ndarray], np.ndarray] | None = None,
                kappa: Callable[[np.ndarray], np.ndarray] | None = None):
        self.tau = params[0]
        self.gamma = params[1]
        self.chi_ = params[2]
        self.mu = mu
        self.domain = domain
        self.domain_resolution = domain_resolution
        self.functionspace = functionspace
        self.display = display
        self.display_name = display_name
        self.order = order
        self.delta = (lambda x: np.ones_like(x)) if delta is None else delta
        self.kappa = (lambda x: np.ones_like(x)) if kappa is None else kappa

    def _setup_domain(self) -> tuple[ufl.Measure, ufl.Measure]:
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
        dx, ds = self._setup_domain()

        # --- Simulation parameters ---
        tau2  = fem.Constant(self.domain, PETSc.ScalarType(self.tau**2))
        gamma = fem.Constant(self.domain, PETSc.ScalarType(self.gamma))
        chi_  = fem.Constant(self.domain, PETSc.ScalarType(self.chi_))
        mu    = fem.Constant(self.domain, PETSc.ScalarType(self.mu))

        # --- Trial and Test Functions --- 
        chi = ufl.TrialFunction(self.functionspace)
        v = ufl.TestFunction(self.functionspace)
        x = self.functionspace.tabulate_dof_coordinates()[:, 0]
        
        delta = fem.Function(self.functionspace, name="delta")
        delta.x.array[:] = self.delta(x)
        kappa = fem.Function(self.functionspace, name="kappa")
        kappa.x.array[:] = self.kappa(x)

        # build forms 
        F = 


        return np.zeros(1), np.zeros(1)


