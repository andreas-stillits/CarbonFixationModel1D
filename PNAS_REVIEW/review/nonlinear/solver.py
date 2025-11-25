import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl 
from dolfinx import fem, mesh, default_scalar_type
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.fem.petsc import NonlinearProblem
import matplotlib.pyplot as plt 
from typing import Callable, Optional, cast


class NonlinearSolver: 
    def __init__(self, 
                params: tuple[float, float, float], # tau, gamma, chi_
                mu: float = 0.0,
                domain: Optional[mesh.Mesh] = None,
                domain_resolution: int = 100,
                functionspace: Optional[fem.FunctionSpace] = None,
                display: bool = False,
                display_name: str = "chi_nonlinear.png",
                order: int = 1,
                delta: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                kappa: Optional[Callable[[np.ndarray], np.ndarray]] = None):
        self.tau = params[0]
        self.gamma = params[1]
        self.chi_ = params[2]
        self.mu = mu
        self.domain_resolution = domain_resolution
        self.domain = mesh.create_interval(MPI.COMM_SELF, self.domain_resolution, [0.0, 1.0]) if domain is None else domain
        self.order = order
        self.functionspace = fem.functionspace(self.domain, ("CG", self.order)) if functionspace is None else functionspace
        self.display = display
        self.display_name = display_name
        self.delta = (lambda x: np.ones_like(x)) if delta is None else delta
        self.kappa = (lambda x: np.ones_like(x)) if kappa is None else kappa

    def _setup_domain(self) -> tuple[ufl.Measure, ufl.Measure]:
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
        tau2  = fem.Constant(self.domain, default_scalar_type(self.tau**2))
        gamma = fem.Constant(self.domain, default_scalar_type(self.gamma))
        chi_  = fem.Constant(self.domain, default_scalar_type(self.chi_))
        mu    = fem.Constant(self.domain, default_scalar_type(self.mu))

        # --- Trial and Test Functions --- 
        chi = fem.Function(self.functionspace, name="chi"); chi = cast(fem.Function, chi)   
        v = ufl.TestFunction(self.functionspace)
        x = self.functionspace.tabulate_dof_coordinates()[:, 0]
        
        delta = fem.Function(self.functionspace, name="delta")
        delta = cast(fem.Function, delta)
        delta.x.array[:] = self.delta(x)

        kappa = fem.Function(self.functionspace, name="kappa")
        kappa = cast(fem.Function, kappa)
        kappa.x.array[:] = self.kappa(x)

        # build forms 
        F = - delta * ufl.inner(ufl.grad(chi), ufl.grad(v)) * dx \
            - tau2 * kappa * (chi - chi_) / (1 + mu * chi) * v * dx \
            + gamma * (1 - chi) * v * ds(1)
        
        problem = NonlinearProblem(F, chi, bcs=[])

        # newton solver 
        solver = NewtonSolver(MPI.COMM_SELF, problem)
        solver.convergence_criterion = "incremental"
        solver.rtol = 1e-8
        solver.atol = 1e-10
        solver.max_it = 25

        ksp = solver.krylov_solver 
        opts = PETSc.Options()
        prefix = ksp.getOptionsPrefix()
        opts[f"{prefix}ksp_type"] = "cg"
        opts[f"{prefix}pc_type"] = "hypre"
        ksp.setFromOptions()

        n_iter, converged = solver.solve(chi)
        assert converged, "Nonlinear solver did not converge"
        
        # extract numpy domain and solution
        self.domain.topology.create_connectivity(0, 1)
        imap = self.domain.topology.index_map(0)
        nloc_vertices = imap.size_local 
        local_vertices = np.arange(nloc_vertices)

        v_dofs = fem.locate_dofs_topological(self.functionspace, 0, local_vertices)
        xv = self.domain.geometry.x[:nloc_vertices, 0].copy()
        uv = chi.x.array[v_dofs].copy()
        seq = np.argsort(xv)
        np_domain = xv[seq]
        np_solution = uv[seq]
        if self.display:
            plt.figure()
            plt.plot(np_domain, np_solution, label="steady solution")
            plt.ylim(0, 1.05)
            plt.xlabel("x")
            plt.ylabel("chi(x)")
            plt.title("Steady State Nonlinear Solution")
            plt.legend()
            plt.savefig(self.display_name)
            plt.close()
        
        return np_domain, np_solution


