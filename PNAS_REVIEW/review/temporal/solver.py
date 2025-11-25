"""

Module to explore the effect of time-dependence on quantities such as gs, Ca and K

"""


from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
import pyvista 
import ufl
from dolfinx import mesh, fem, plot, default_scalar_type
from dolfinx.fem.petsc import create_matrix, create_vector, assemble_vector, assemble_matrix
from typing import Callable, Optional, cast
import functools
from review.steady.solver import SteadySolver

class TemporalSolver:
    def __init__(self, 
                 params: tuple[float, float, float], # [tau, gamma, chi_]
                 timing: tuple[float, float, float] = (0.0, 1.0, 0.01),
                 domain: Optional[mesh.Mesh] = None, 
                 domain_resolution: int = 100, 
                 functionspace: Optional[fem.FunctionSpace] = None,
                 animate: bool = False,
                 animation_name: str = "chi_time.gif",
                 order: int = 1,
                 update_delta: Optional[Callable[[np.ndarray, float], np.ndarray]] = None,
                 update_kappa: Optional[Callable[[np.ndarray, float], np.ndarray]] = None,
                 update_stomata: Optional[Callable[[np.ndarray, float], np.ndarray]] = None,
                 update_atmospheric: Optional[Callable[[np.ndarray, float], np.ndarray]] = None,
                 initial_condition: Optional[np.ndarray] = None):
        self.tau = params[0]
        self.gamma = params[1]
        self.chi_ = params[2]
        self.t_start = timing[0]
        self.t_end = timing[1]
        self.dt = timing[2]
        self.domain_resolution = domain_resolution
        self.domain = mesh.create_interval(MPI.COMM_SELF, domain_resolution, [0.0, 1.0]) if domain is None else domain
        self.order = order
        self.functionspace = fem.functionspace(self.domain, ("CG", self.order)) if functionspace is None else functionspace
        self.animate = animate
        self.animation_name = animation_name
        self.update_delta = (lambda x, t: np.ones_like(x)) if update_delta is None else update_delta
        self.update_kappa = (lambda x, t: np.ones_like(x)) if update_kappa is None else update_kappa
        self.update_stomata = (lambda x, t: np.ones_like(x)) if update_stomata is None else update_stomata
        self.update_atmospheric = (lambda x, t: np.ones_like(x)) if update_atmospheric is None else update_atmospheric
        if initial_condition is None:
            steady_solver = SteadySolver(params, 
                                         domain=self.domain, 
                                         domain_resolution=self.domain_resolution, 
                                         functionspace=self.functionspace, 
                                         order=self.order, 
                                         delta = functools.partial(self.update_delta, t=0.0),
                                         kappa = functools.partial(self.update_kappa, t=0.0))
            _, chi_steady = steady_solver.solve()
            self.initial_condition = chi_steady


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
    

    def _setup_plotting(self, chi_n: fem.Function) -> tuple[np.ndarray, pyvista.UnstructuredGrid, pyvista.Plotter]:
        grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(self.functionspace))
        xcoords = grid.points[:, 0].copy()
        grid.point_data["chi"] = chi_n.x.array.copy()  
        grid.points = np.c_[xcoords, 
                            grid.point_data["chi"], 
                            np.zeros_like(xcoords)]  # x, chi, 0 

        plotter = pyvista.Plotter(window_size=[500, 400])
        plotter.open_gif(self.animation_name, fps=10)
        actor = plotter.add_mesh(
            grid,
            show_edges=True,
            lighting=False,
            show_scalar_bar=False,
            clim=(0.0, 1.0),
            line_width=4
        )

        homogeneous = grid.copy(deep=True)
        homogeneous.points = np.c_[xcoords, 
                                   self.initial_condition, 
                                   np.zeros_like(xcoords)]
        homogeneous_actor = plotter.add_mesh(
            homogeneous,
            color="black",
            show_edges=False,
            lighting=False,
            line_width=2
        )

        vmin, vmax = 0.0, 1.0 
        bounds = (vmin, vmax, vmin, vmax, 0.0, 0.0)
        cube = plotter.show_grid( # type: ignore[reportCallIssue]
            bounds=bounds,
            location="outer",
            xtitle=r"$\lambda$ (normalized depth)",
            ytitle=r"$\chi$ (normalized concentration)",
            ticks="both",
            font_size=2,
            grid="both"
        )
        cube.SetLabelOffset(4)
        
        plotter.view_xy() # type: ignore[reportCallIssue]
        plotter.set_focus((0.5, 0.5, 0.0)) # type: ignore[reportArgumentType]
        plotter.set_position((0.5, 0.5, 10.0)) # type: ignore[reportArgumentType]
        plotter.camera.parallel_projection = True
        plotter.camera.SetParallelScale(0.5)
        plotter.camera.zoom(0.8)
        plotter.write_frame()

        return xcoords, grid, plotter

    
    def solve(self) -> tuple[np.ndarray, np.ndarray]:
        # --- get measures and set up domain mesh and functionspace if not supplied ---
        dx, ds = self._setup_domain()
        
        # --- simulation parameters ---
        tau2  = fem.Constant(self.domain, default_scalar_type(self.tau**2))
        gamma = fem.Constant(self.domain, default_scalar_type(self.gamma))
        chi_  = fem.Constant(self.domain, default_scalar_type(self.chi_))
        
        # --- Trial and Test Functions ---
        chi = ufl.TrialFunction(self.functionspace)
        v   = ufl.TestFunction(self.functionspace)
        
        # --- time stepping parameters --- 
        dt        = fem.Constant(self.domain, default_scalar_type(self.dt))
        num_steps = int(np.ceil((self.t_end - self.t_start) / self.dt))
        t         = self.t_start

        # --- initial condition ---
        chi_n            = fem.Function(self.functionspace, name="chi_n")
        chi_n            = cast(fem.Function, chi_n)
        x                = self.functionspace.tabulate_dof_coordinates()[:, 0]
        chi_initial = np.interp(x, x, self.initial_condition, left=self.initial_condition[0], right=self.initial_condition[-1])
        chi_n.x.array[:] = chi_initial       
        chi_n.x.scatter_forward()

        # --- time dependent coefficients ---
        delta_t       = fem.Function(self.functionspace, name="delta_t"); delta_t = cast(fem.Function, delta_t)
        kappa_t       = fem.Function(self.functionspace, name="kappa_t"); kappa_t = cast(fem.Function, kappa_t)
        stomata_t     = fem.Function(self.functionspace, name="stomata_t"); stomata_t = cast(fem.Function, stomata_t)
        atmospheric_t = fem.Function(self.functionspace, name="atmospheric_t"); atmospheric_t = cast(fem.Function, atmospheric_t)

        # --- variational forms for backwards euler ---
        d, k, s, ca = delta_t, kappa_t, stomata_t, atmospheric_t # aliases for readability in forms
        
        a = chi * v * dx \
            +dt * tau2 * k * chi * v * dx \
            +dt * d * ufl.inner(ufl.grad(chi), ufl.grad(v)) * dx \
            +dt * gamma * s * chi * v * ds(1) # left boundary at lambda = 0
        
        L = chi_n * v * dx \
            +dt * tau2 * k * chi_ * v  * dx \
            +dt * gamma * s * ca * v * ds(1) # left boundary at lambda = 0

        a = fem.form(a)
        L = fem.form(L) 

        # precreate PETSc objects with correct sparsity pattern 
        A = create_matrix(a) # type: ignore[reportArgumentType]
        b = create_vector(L) # type: ignore[reportArgumentType]
        ksp = PETSc.KSP().create(MPI.COMM_SELF) # type: ignore[reportAttributeAccessIssue]
        ksp.setType("cg")          # SPD for this model (with Dirichlet or pure Neumann + mass term)
        ksp.getPC().setType("hypre")
        ksp.setTolerances(rtol=1e-10, atol=1e-12, max_it=500)
        chi_new = fem.Function(self.functionspace, name="chi_new"); chi_new = cast(fem.Function, chi_new)

        # --- create animation objects if requested ---
        if self.animate:
            xcoords, grid, plotter = self._setup_plotting(chi_n)

        # --- Extract time series of rescaled assimilation rate ---        
        times = [] 
        alphas = []

        # START INTEGRATION LOOP

        for n in range(num_steps):
            t += self.dt

            # update time-dependent coefficients
            delta_t.x.array[:] = self.update_delta(x, t); delta_t.x.scatter_forward()
            kappa_t.x.array[:] = self.update_kappa(x, t); kappa_t.x.scatter_forward()
            stomata_t.x.array[:] = self.update_stomata(x, t); stomata_t.x.scatter_forward()
            atmospheric_t.x.array[:] = self.update_atmospheric(x, t); atmospheric_t.x.scatter_forward()

            # reassemble A and b due to time-dependence 
            A.zeroEntries() 
            assemble_matrix(A, a)
            A.assemble()

            with b.localForm() as b_local:
                b_local.set(0.0)
            assemble_vector(b, L)

            # Solve A chi_n = b 
            ksp.setOperators(A)
            ksp.solve(b, chi_new.x.petsc_vec)
            
            # update for next time step
            chi_new.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD) # type: ignore[reportAttributeAccessIssue]
            chi_n.x.array[:] = chi_new.x.array
            chi_n.x.scatter_forward() 

            # --- update plotting --- 
            if self.animate:
                grid.point_data["chi"][:] = chi_new.x.array # type: ignore[reportArgumentType]
                grid.points[:,:] = np.c_[xcoords, # type: ignore[reportArgumentType]
                                         chi_new.x.array,
                                         np.zeros_like(xcoords)]  #
                plotter.write_frame()
            
            # --- extract rescaled assimilation rate at this time step ---
            times.append(t)
            alpha = fem.assemble_scalar(fem.form(tau2 * kappa_t * (chi_new - chi_)*dx)) # type: ignore[reportArgumentType]
            alphas.append(alpha)
            # END INTEGRATION LOOP
        
        for item in [A, b, ksp]:
            item.destroy()
        if self.animate:
            plotter.close()

        return np.array(times), np.array(alphas)

        